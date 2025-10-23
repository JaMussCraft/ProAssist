#!/bin/bash
#SBATCH --job-name=finetune_smolvlm2_500M_wtag_9 # CUSTOMIZE name
#SBATCH --account=beto-delta-gpu
#SBATCH --partition=gpuA100x4        
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1               # Request 1 A100 GPU
#SBATCH --cpus-per-task=8               # 8 CPU cores
#SBATCH --mem=100G                       # 100GB RAM
#SBATCH --time=18:00:00                 # CUSTOMIZE Max runtime
#SBATCH --output=./finetune_smolvlm2_500M_wtag_9/finetune_smolvlm2_500M_wtag_9_%j.out # CUSTOMIZE
#SBATCH --error=./finetune_smolvlm2_500M_wtag_9/finetune_smolvlm2_500M_wtag_9_%j.err  # CUSTOMIZE

# Load necessary modules (if needed on Delta)
# module load anaconda3_gpu  # Uncomment if conda isn't in your environment

# VERSION 9 computes loss for w2t (predicts EOS), talk positions (predicts any token other than EOS - next next token), and assistant tokens
# AND use end-of-utterance for w2t token
# AND use batch-dynamic, normalized, inverse frequency weighted cross entropy loss instead of static w2t_loss_weight 

# RESULTS: both talk and no-talk positions still diverge, the general trend is still decreasing prob; but prob reached a lower lowest point
# grad_norm is on the unstable side; weighted training loss is lower like around 0.5
# Streaming inference results: tend to speak in the beginning and stop speaking as conversation gets longer
# assistant responses are conversational, but not very good. responses make no sense sometimes


# Activate conda environment
source ~/.bashrc  # or wherever conda is initialized
conda activate smolvlm

# Set Python path
export PYTHONPATH=/u/swong2/ProAssist:$PYTHONPATH

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Run the fine-tuning script
python ../finetune_smolvlm.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
    --use_lora true \
    --freeze_vision false \
    --data_root_dir /projects/beto/proassist_data/processed_data/ \
    --train_datasets wtag/dialog-klg-sum_train_L2048_I1 \
    --eval_datasets wtag/dialog-klg-sum_val_L2048_I1 \
    --use_4_1_aspect_ratio true \
    --frame_sampling_ratio 0.1 \
    --w2t_frame_sampling_rate 1 \
    --output_dir ./finetune_smolvlm2_500M_wtag_9 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 5 \
    --gradient_checkpointing true \
    --dataloader_num_workers 6 \
    --dataloader_prefetch_factor 4 \
    --eval_steps 150 \
    --logging_steps 150 \
    --use_inverse_freq_weighting true \
    --use_end_of_utterance_for_w2t true

echo "Job finished at: $(date)"