#MATHLlama
PROMPT1="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>."
#MATHQwen
PROMPT2="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n\boxed{{your answer here}}\n</answer>."
#GSM8K
PROMPT3="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>\nYour reasoning here\n</think>\n<answer>\n answer here \n</answer>. The reasoning process Note that respond by English, NOT use other languages."

export CUDA_VISIBLE_DEVICES=0,1,2,3
NOFGPU=4
MODEL=qwen            # qwen or llama
DATA=MATH             # MATH or GSM8K
# ==========================


case $MODEL in
  qwen)
    REF_MODEL="Qwen/Qwen2.5-3B-Instruct"
    MAIN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    ;;
  llama)
    REF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
    MAIN_MODEL="meta-llama/Llama-3.2-1B-Instruct"
    ;;
  *)
    echo "Invalid MODEL: $MODEL"
    exit 1
    ;;
esac


case $DATA in
  MATH)
    PDBATCH_SIZE=4
    MAX_LENGTH=2048
    DATA_NAME="dataset/math12k"
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
    PDBATCH_SIZE=8
    MAX_LENGTH=1024
    DATA_NAME="dataset/gsm8k"
    PROMPT_NO=3
    REWARD=rule_based_accuracy
    ;;
  *)
    echo "Invalid DATA: $DATA"
    exit 1
    ;;
esac

#Qwen/Qwen2.5-3B-Instruct

PROMPT_VAR="PROMPT${PROMPT_NO}"
SYSTEM_PROMPT="${!PROMPT_VAR}"
for i in {1..100}; do
    CANDIDATE_NAME="unir_${MODEL}_${DATA}_v${i}"
    CANDIDATE_DIR="run/${CANDIDATE_NAME}"
    if [ ! -d "$CANDIDATE_DIR" ]; then
        NAME=$CANDIDATE_NAME
        break
    fi
done
mkdir -p run/"$NAME"
srun --export=ALL,ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --main_process_port 8614 \
  --num_processes $NOFGPU \
  src/unir/train_vision.py \
  --dataset_name "$DATA_NAME" \
  --ref_name_or_path "$REF_MODEL" \
  --model_name_or_path  "$MAIN_MODEL" \
  --use_peft False \
  --config recipes/unir.yaml \
  --output_dir run/"$NAME" \
  --run_name test_vision \
  --num_generations 8 \
  --per_device_eval_batch_size $PDBATCH_SIZE \
  --per_device_train_batch_size $PDBATCH_SIZE \
  --gradient_accumulation_steps $((8/NOFGPU))  \
  --gradient_checkpointing false \
  --max_completion_length 512 \
  --max_steps 1000 \
  --save_steps 100 \
  --beta 0 \
  --system_prompt "$SYSTEM_PROMPT" \
  --reward_funcs "$REWARD" \
  --reward_weights 1.0 \
  >  run/"$NAME"/train_log.log 2>&1