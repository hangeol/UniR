CHECKPOINT_ROOT="run/test2"
CONFIG="recipes/unir.yaml"
SCRIPT="src/unir/evaluate.py"
mkdir -p "$CHECKPOINT_ROOT/output_log"
ckpt_num=500
dataset_index=1

#['aime24', 'math500',  'minerva',  'olympiadbench', 'gsm8k'] #dataset index
#MATHLlama
PROMPT1="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>."
#MATHQwen
PROMPT2="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n\boxed{{your answer here}}\n</answer>."
#GSM8K
PROMPT3="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>. The reasoning process Note that respond by English, NOT use other languages."


MODEL=qwen # qwen or llama
DATA=MATH # MATH or GSM8K
MODEL_SIZE=3b            # 3b, 7b, 8b 또는 14b





case $MODEL in
  qwen)
    MAIN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    case $MODEL_SIZE in
      3b)
        REF_MODEL="Qwen/Qwen2.5-3B-Instruct"
        EVAL_SIZE=8
        ;;
      7b)
        REF_MODEL="Qwen/Qwen2.5-7B-Instruct"
        EVAL_SIZE=4
        ;;
      14b)
        REF_MODEL="Qwen/Qwen2.5-14B-Instruct"
        EVAL_SIZE=2
        ;;
      *)
        echo "Invalid MODEL_SIZE: $MODEL_SIZE"
        exit 1
        ;;
    esac
    ;;
  llama)
    MAIN_MODEL="meta-llama/Llama-3.2-1B-Instruct"
    case $MODEL_SIZE in
      3b)
        REF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
        EVAL_SIZE=8
        ;;
      8b)
        REF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
        EVAL_SIZE=4
        ;;
      *)
        echo "Invalid MODEL_SIZE: $MODEL_SIZE"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Invalid MODEL: $MODEL"
    exit 1
    ;;
esac


case $DATA in
  MATH)
    MAX_LENGTH=2048
    case $MODEL in
      qwen)
        PROMPT_NO=2
        REWARD=boxed_reward
        ;;
      llama)
        PROMPT_NO=1
        REWARD=tag_based_reward
        ;;
      *)
        echo "Invalid MODEL: $MODEL"
        exit 1
        ;;
    esac
    ;;
  GSM8K)
    MAX_LENGTH=1024
    PROMPT_NO=3
    REWARD=rule_based_accuracy
    ;;
  *)
    echo "Invalid DATA: $DATA"
    exit 1
    ;;
esac



for ckpt_num in $(seq 10 10 20); do
  for dataset_index in 0 1 2 3; do
      LOG_PATH="${CHECKPOINT_ROOT}/output_log/checkpoint-${ckpt_num}_dataset_${dataset_index}"
      python "$SCRIPT" \
      --config "$CONFIG" \
      --dataset_config default \
      --num_generations 1 \
      --per_device_eval_batch_size $EVAL_SIZE \
      --max_completion_length $MAX_LENGTH \
      --gradient_checkpointing false \
      --model_name_or_path "$MAIN_MODEL" \
      --ref_name_or_path "$REF_MODEL" \
      --eval_checkpoint "$CHECKPOINT_ROOT"/checkpoint-$ckpt_num \
      --use_vllm false \
      --output_dir "$LOG_PATH" \
      --run_name unir_test \
      --reward_funcs "$REWARD" \
      --reward_weights 1.0 \
      --temperature 0.0 \
      --beta 0.0 \
      --dataset_index $dataset_index \
      --system_prompt "$SYSTEM_PROMPT" \
      > "${LOG_PATH}.log" 2>&1
    done
done