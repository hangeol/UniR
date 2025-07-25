import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from rewards import (rulebased_correct_reward_func, boxed_reward_fn_qwen , boxed_reward_fn_llama)
from unir.utils import get_tokenizer
from unir.utils.callbacks import get_callbacks
from trl import TrlParser, get_peft_config
from peft import PeftModel, PeftConfig
from unir.trainer.UniRTrainer import UniRGRPOTrainer
from unir.configs import UniRGRPOConfig,UniRGRPOModelConfig,GRPOScriptArguments
import json
import textwrap
from dotenv import load_dotenv
load_dotenv()
DATASET_NAME = {
    'aime24':'HuggingFaceH4/aime_2024', 
    'math500':'HuggingFaceH4/MATH-500', 
    'amc23':'knoveleng/AMC-23',
    'olympiadbench':'knoveleng/OlympiadBench', 
    'minerva':'knoveleng/Minerva-Math', 
    'gsm8k': 'openai/gsm8k'
}



def evaluate(script_args, training_args, model_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(training_args.get_process_log_level())

    # Seed
    set_seed(training_args.seed)

    # Load dataset
    training_args.eval_strategy = "yes"
    training_args.log_completions = False
    dataset_list = ['aime24', 'math500',  'minerva',  'olympiadbench', 'gsm8k']

    dataset_name = dataset_list[model_args.dataset_index]
    script_args.dataset_name = DATASET_NAME[dataset_name]
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def make_conversation(example):
        prompt = []
        try:
            question = example["problem"]
        except:
            question = example["question"]
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": question})
        return {"prompt": prompt}
        
    def make_conversation_llama(example):
        try:
            question = example["problem"]
        except:
            question = example["question"]
        prompt =(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>      \n\n"
            f"{training_args.system_prompt}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>   \n\n"
            f"{question}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return {"prompt": prompt}

    print("model_name: ",model_args.model_name_or_path.lower())
    if "llama" in model_args.model_name_or_path.lower():
        print("llama dataset mapping")
        dataset = dataset.map(make_conversation_llama)
    else:
        dataset = dataset.map(make_conversation)


    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Load tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        

    # Prepare reward functions
    REWARD_FUNCS_REGISTRY = {
        "tag_based_reward" : boxed_reward_fn_llama,
        "boxed_reward" : boxed_reward_fn_qwen,
        "rule_based_accuracy": rulebased_correct_reward_func,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Model kwargs
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=True,
    )
    training_args.model_init_kwargs = model_kwargs
    load_checkpoint = model_args.eval_checkpoint
    # Load model
    print("Load checkpoint from:", load_checkpoint)
    if load_checkpoint != "None": 
        if model_args.use_peft :
            training_args.lora_weight = load_checkpoint
            peft_config = PeftConfig.from_pretrained(load_checkpoint)
            base_model = model_args.model_name_or_path
        else : 
            base_model = os.path.join(load_checkpoint,"model")
            peft_config = get_peft_config(model_args)
    else : 
        base_model = model_args.model_name_or_path 
        peft_config = get_peft_config(model_args)
    
    if model_args.use_unir==False:
        ref_model = None
    else:
        ref_model = model_args.ref_name_or_path
    print(f"base model from {base_model}")
    print(f"ref model from {ref_model}")
    test_data = dataset.get('test', dataset.get('train'))
    trainer = UniRGRPOTrainer(
        model=base_model,
        ref_model=ref_model,
        use_unir=model_args.use_unir,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=None,  # No training
        eval_dataset=test_data ,
        peft_config=peft_config,
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # Evaluate
    print("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(test_data)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    custom_filename = f"eval_results_{os.path.basename(load_checkpoint)}.json"
    save_path = os.path.join(training_args.output_dir, "output_log", custom_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, UniRGRPOConfig, UniRGRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    evaluate(script_args, training_args, model_args)
