# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig,UniRGRPOModelConfig,GRPOScriptArguments

from rewards import (rulebased_correct_reward_func, boxed_reward_fn_qwen , boxed_reward_fn_llama)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer.UniRTrainer import UniRGRPOTrainer
#from trl.trainer.grpo_trainer import GRPOTrainer as trlGRPOTrainer
import wandb
import json
from datasets import load_from_disk
import pyarrow as pa
import pyarrow.ipc as ipc


os.environ["HF_TOKEN"] = "your token here"


logger = logging.getLogger(__name__)





def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    #wandb.init(project="R-GRPO", name=training_args.run_name)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "tag_based_reward" : boxed_reward_fn_llama,
        "boxed_reward" : boxed_reward_fn_qwen,
        "rule_based_accuracy": rulebased_correct_reward_func,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
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
        prompt = textwrap.dedent(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>      

            {training_args.system_prompt}
            <|eot_id|><|start_header_id|>user<|end_header_id|>   

            {question}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
        return {"prompt": prompt}

    print("model_name_for_dataset_map",model_args.model_name_or_path.lower())
    
    if "llama" in model_args.model_name_or_path.lower():
        dataset = dataset.map(make_conversation_llama)
    else:
        dataset = dataset.map(make_conversation)


    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    training_args.optim = "paged_adamw_8bit" 
    trainer = UniRGRPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=model_args.ref_name_or_path, 
        use_unir=model_args.use_unir,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["UniR"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)




if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, UniRGRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
