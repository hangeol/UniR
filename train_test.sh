CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --main_process_port 8610 \
  --num_processes=1 \
  src/unir/train.py \
  --dataset_name data/math12k \
  --ref_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --config recipes/unir.yaml \
  --output_dir run/test2 \
  --run_name test \
  --num_generations 2 \
  --per_device_eval_batch_size 2 \
  --per_device_train_batch_size  2 \
  --gradient_accumulation_steps 2  \
  --max_completion_length 100 \
  --max_steps 100 \
  --save_steps 10 \
  --beta 0 \
  --gradient_checkpointing false \
  --system_prompt  "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>."  \
  --reward_funcs tag_based_reward \
  --reward_weights 1.0 \
  > test.log 2>&1