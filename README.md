# Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs

This repository hosts the code and datasets for the **UniR** project, accompanying the paper [*Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs*](https://arxiv.org/abs/). 


![Main Figure](assets/overall.png)
Overview: UniR is a lightweight, plug-and-play reasoning module that enables modular, reward-driven reasoning enhancements for any frozen LLM, achieving strong generalization and composability without retraining the backbone.

<!-- ## Resources -->

<!-- ### Models
- [Open-RS1](https://huggingface.co/knoveleng/Open-RS1)
- [Open-RS2](https://huggingface.co/knoveleng/Open-RS2)
- [Open-RS3](https://huggingface.co/knoveleng/Open-RS3)
- Additional models in training: [knoveleng/OpenRS-GRPO](https://huggingface.co/knoveleng/OpenRS-GRPO/commits/main), [quyanh/OpenRS-GRPO](https://huggingface.co/quyanh/OpenRS-GRPO/commits/main) -->

## Datasets
Datasets can be downloaded from the below links and should be placed under the` ./data` directory.
- [GSM8k](https://huggingface.co/datasets/openai/gsm8k)
- [Math12k](https://huggingface.co/datasets/hiyouga/math12k)

<!-- ### Collection
- [Open-RS Collection](https://huggingface.co/collections/knoveleng/open-rs-67d940abc201a7e7f252ca4e) -->

## Installation
Our codebase is based on [TRL](https://huggingface.co/docs/trl/index) and [open-rs](https://github.com/knoveleng/open-rs) for training. Run the following to set up the environment:

Set up a virtual environment with Python 3.11:
```
conda create -n unir python=3.11
conda activate unir
pip install --upgrade pip
pip install vllm==0.7.2 
pip install setuptools
pip install flash-attn --no-build-isolation
pip install bitsandbytes
pip install -e ".[dev]"
pip install e2b-code-interpreter peft datasets pylatexenc
```

## Training

Train models using a YAML config with 2 GPUs (set `num_processes=2`):

- GSM8K (Llama)
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --main_process_port 6667 \
  --num_processes=2 \
  train.py \
  --config recipes/UniR.yaml \
  --dataset_name dataset/gsm8k \
  --dataset_config default \
  --output_dir run/GSM8k-llama-backbone3b_reasoning1b \
  --run_name GSM8k-llama-backbone3b_reasoning1b \
  --ref_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --num_generations 8 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --max_completion_length 1024 \
  --max_steps 1000 \
  --save_steps 100 \
  --beta 0.0 \
  --system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>. The reasoning process Note that respond by English, NOT use other languages." \
  --reward_funcs rulebased_correct \
  --reward_weights 1.0
```

- Math12k (Qwen)
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --main_process_port 6667 \
  --num_processes=2 \
  train.py \
  --config recipes/rgrpo.yaml \
  --dataset_name dataset/math12k \
  --dataset_config default \
  --output_dir run/GSM8k-qwen-backbone3b_reasoning05b \
  --run_name GSM8k-qwen-backbone3b_reasoning05b \
  --ref_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --num_generations 8 \
  --per_device_eval_batch_size 4 \
  --per_device_train_batch_size  4 \
  --max_completion_length 2048 \
  --max_steps 1000 \
  --save_steps 100 \
  --beta 0.0 \
  --system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n\boxed{{your answer here}}\n</answer>." \
  --reward_funcs boxed \
  --reward_weights 1.0
```

## Evaluation

For single-GPU setups:

- GSM8K (Llama)
```bash
LOG_PATH="$CHECKPOINT_ROOT/output_log/ref3b_trained1b-1000.log"
python eval.py \
--config recipes/UniR.yaml \
--dataset_name dataset/gsm8k \
--dataset_config default \
--do_eval true \
--do_train false \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 1 \
--num_generations 1 \
--vllm_max_model_len 1024 \
--max_completion_length 1024 \
--gradient_checkpointing false \
--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
--ref_name_or_path meta-llama/Llama-3.2-3B-Instruct \
--eval_checkpoint GSM8k-llama-backbone3b_reasoning1b/checkpoint-1000 \
--output_dir run/GSM8k-llama-backbone3b_reasoning1b \
--run_name GSM8k-llama-backbone3b_reasoning1b \
--reward_funcs rulebased_correct \
--reward_weights 1.0 \
--temperature 0.0 \
--beta 0.0 \
--system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>. The reasoning process Note that respond by English, NOT use other languages." \
> "$LOG_PATH" 2>&1
```

- Math 500 (Qwen) 
```bash

```

<!-- 
> **Important**: Set `max_model_length=32768` to match `max_new_tokens`, or `lighteval` will fail.

For multi-GPU evaluation with data parallelism:
```bash
NUM_GPUS=4
MODEL=knoveleng/Open-RS3
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/open_r1/evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"
```

Alternatively, use the evaluation script:
```bash
sh eval.sh
```
Modify tasks in `eval.sh` (line 8) as needed. -->
<!-- 
### Performance Highlights
- **Open-RS1**: 53.0% avg. score
- **Open-RS2**: 55.7% avg. score, 80.0% on AMC23
- **Open-RS3**: 56.3% avg. score, 46.7% on AIME24 (outperforms `o1-preview` at 44.6%)
- Competitive MATH-500 scores; Minerva lags behind 7B models.

![Performance Metrics](assets/performances.png)

### Cost Efficiency
Our approach uses 7,000 samples (42,000 total outputs) and costs ~$42 on 4x A40 GPUs in 24 hours, compared to:
- 7B models: `Qwen2.5-7B-SimpleRL` ($1,633), `Eurus-2-7B-PRIME` ($1,088)
- 1.5B models: `DeepScaleR-1.5B-Preview` ($3,629), `Still-3-1.5B-Preview` ($2,268)

![7B Model Costs](assets/costs-7b.png)  
![1.5B Model Costs](assets/costs-1.5b.png) -->

## Acknowledgements
Thanks to the Hugging Face team for their [open-r1](https://github.com/huggingface/open-r1) and  [open-rs](https://github.com/knoveleng/open-rs) project.

## Citation
If this project aids your work, please cite it as:
<!-- ```
@misc{dang2025reinforcementlearningreasoningsmall,
      title={Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't}, 
      author={Quy-Anh Dang and Chris Ngo},
      year={2025},
      eprint={2503.16219},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.16219}, 
}
``` -->
