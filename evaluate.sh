CHECKPOINT_ROOT="run/test"
CONFIG="recipes/unir.yaml"
SCRIPT="src/unir/evaluate.py"
mkdir -p "$CHECKPOINT_ROOT/output_log"
ckpt_num=500
dataset_index=1

#['aime24', 'math500',  'minerva',  'olympiadbench', 'gsm8k'] #dataset index

for ckpt_num in $(seq 10 100 1000); do
  for dataset_index in 0 1 2 3; do
      LOG_PATH="${CHECKPOINT_ROOT}/output_log/checkpoint-${ckpt_num}_dataset_${dataset_index}"
      python "$SCRIPT" \
      --config "$CONFIG" \
      --dataset_config default \
      --num_generations 1 \
      --per_device_eval_batch_size 1 \
      --max_completion_length 2048 \
      --gradient_checkpointing false \
      --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
      --ref_name_or_path Qwen/Qwen2.5-3B-Instruct \
      --eval_checkpoint "$CHECKPOINT_ROOT"/checkpoint-$ckpt_num \
      --use_vllm false \
      --output_dir "$LOG_PATH" \
      --run_name unir_test \
      --reward_funcs boxed_reward \
      --reward_weights 0.5 \
      --temperature 0.0 \
      --beta 0.0 \
      --dataset_index $dataset_index \
      --system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n\boxed{{your answer here}}\n</answer>." \
      > "${LOG_PATH}.log" 2>&1
    done
done