#!/bin/bash

# SmolVLM ProAssist Fine-tuning Script
# This script runs the SmolVLM fine-tuning on ProAssist dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:/u/swong2/ProAssist"

# Data configuration
DATA_ROOT_DIR="/projects/beto/swong2/proassist_data/processed_data"
TRAIN_DATASETS="wtag/dialog-klg_train_L0_I1"
EVAL_DATASETS="wtag/dialog-klg_val_L0_I1"

# Model configuration
MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"
OUTPUT_DIR="./smolvlm-proassist-$(date +%Y%m%d-%H%M%S)"

# Training configuration
BATCH_SIZE=2
GRAD_ACCUMULATION=8
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=8192
FRAMES_PER_SAMPLE=3

echo "Starting SmolVLM ProAssist Fine-tuning"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_ROOT_DIR"
echo "Train datasets: $TRAIN_DATASETS"
echo "Eval datasets: $EVAL_DATASETS"
echo "======================================"

python mmassist/train/finetune_smolvlm.py \
    --model_name_or_path $MODEL_NAME \
    --use_lora true \
    --use_qlora false \
    --freeze_vision false \
    --data_root_dir $DATA_ROOT_DIR \
    --train_datasets $TRAIN_DATASETS \
    --eval_datasets $EVAL_DATASETS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --use_4_1_aspect_ratio true \
    --frames_per_sample $FRAMES_PER_SAMPLE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --logging_steps 25 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --optim paged_adamw_8bit \
    --bf16 true \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "down_proj,o_proj,k_proj,q_proj,gate_proj,up_proj,v_proj" \
    --use_dora false

echo "Training completed! Model saved to: $OUTPUT_DIR"
