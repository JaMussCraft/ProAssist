#!/bin/bash
#SBATCH --job-name=mllm-train
##SBATCH --account=
#SBATCH --partition=q1
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1991800
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/project/proactive-assist/slurm_logs/%j-12node-train.log
source activate mm
cat /etc/hosts
cd /data/home/imzyc/project/proactive-assist/


MAX_SEQ_LEN=4096
USE_IMG_CLS_TOKEN=True
IMG_PATCH_NUM=2
IMG_SEP_TOKEN="none"
if [ "$IMG_SEP_TOKEN" = ";" ]; then
    SEP_NAME=SEP_SEMICOLON
    SEP_SUFFIX=+SEP
elif [ "$IMG_SEP_TOKEN" = "none" ]; then
    SEP_NAME=NOSEP
    SEP_SUFFIX=""
else
    SEP_NAME=SEP_$IMG_SEP_TOKEN
    SEP_SUFFIX=+SEP
fi

if [ "$USE_IMG_CLS_TOKEN" = "True" ]; then
    NUM_IMG_TOKENS=$(($IMG_PATCH_NUM * $IMG_PATCH_NUM + 1))
else
    NUM_IMG_TOKENS=$(($IMG_PATCH_NUM * $IMG_PATCH_NUM))
fi
DATE=$(date '+%Y%m%d')
LR=2e-4
BATCH_SIZE_PER_DEVICE=4
GRAD_ACCUM_STEPS=1
NUM_GPUS_PER_NODE=8
NUM_WORKERS=2
BATCH_SIZE=$(($BATCH_SIZE_PER_DEVICE * $NUM_GPUS_PER_NODE * $GRAD_ACCUM_STEPS * $SLURM_JOB_NUM_NODES))
EPOCHS=4
NFSR=0.1

# ADD_KNOWLEDGE=""
ADD_KNOWLEDGE="-klg"
# MIX_TRAIN_KLG=""
MIX_TRAIN_KLG=mix
DATA_TYPE=dialog${ADD_KNOWLEDGE}-sum

USE_BINARY_HEAD=False
USE_MLP_HEAD=True
BINARY_WEIGHT=0.1
if [ "$USE_BINARY_HEAD" = "True" ]; then
    if [ "$USE_MLP_HEAD" = "True" ]; then
        BINARY_FLAG="-w2t_head_mlp_w${BINARY_WEIGHT}"
        BINARY_TYPE="mlp"
    else
        BINARY_FLAG="-w2t_head_w${BINARY_WEIGHT}"
        BINARY_TYPE="linear"
    fi
    FT_MODULES="mm_projector,binary_decision_head"
else
    BINARY_FLAG=""
    BINARY_TYPE="linear"
    FT_MODULES="mm_projector"
fi

USE_POSE=False
if [ "$USE_POSE" = "True" ]; then
    POSE_FLAG="-pose"
else
    POSE_FLAG=""
fi

# if [ "$IMG_PATCH_NUM" = "3" ]; then
#     ZERO_CONFIG=zero2_offload
# elif [ "$IMG_PATCH_NUM" = "5" ]; then
#     ZERO_CONFIG=zero2_offload
# else
#     ZERO_CONFIG=zero2
# fi
# ZERO_CONFIG=zero2
ZERO_CONFIG=zero2_offload


EXP_NAME=${DATE}-L${MAX_SEQ_LEN}-I${NUM_IMG_TOKENS}-ep${EPOCHS}-${SEP_NAME}-nr${NFSR}${ADD_KNOWLEDGE}${MIX_TRAIN_KLG}${BINARY_FLAG}${POSE_FLAG}-1s-lora-bs${BATCH_SIZE}
# EXP_NAME=${DATE}-debug-5e7
echo $EXP_NAME

OUTDIR=${DATA_ROOT_DIR}/proact_exps/$EXP_NAME
mkdir -p $OUTDIR


# Stage 1 datasets: mixed
TRAIN_DATASETS="ego4d/narration_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
TRAIN_DATASETS="$TRAIN_DATASETS,sthsthv2/narration_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@10"
TRAIN_DATASETS="$TRAIN_DATASETS,llava/caption_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}@2"
TRAIN_DATASETS="$TRAIN_DATASETS,egoobjects/detection_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}@20"

# Stage 1 datasets: summary
for DATASET in ego4d holoassist epickitchens egoexolearn wtag assembly101; do
    TRAIN_DATASETS="$TRAIN_DATASETS,$DATASET/summary_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
done

# Stage 2 datasets 
if [ "$MIX_TRAIN_KLG" = "True" ]; then
    TRAIN_DATASETS="$TRAIN_DATASETS,ego4d/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,holoassist/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,epickitchens/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,egoexolearn/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,wtag/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@3"
    TRAIN_DATASETS="$TRAIN_DATASETS,assembly101/dialog-klg-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"

    TRAIN_DATASETS="$TRAIN_DATASETS,ego4d/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,holoassist/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,epickitchens/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,egoexolearn/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
    TRAIN_DATASETS="$TRAIN_DATASETS,wtag/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@3"
    TRAIN_DATASETS="$TRAIN_DATASETS,assembly101/dialog-sum_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
else
    TRAIN_DATASETS="$TRAIN_DATASETS,ego4d/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
    TRAIN_DATASETS="$TRAIN_DATASETS,holoassist/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
    TRAIN_DATASETS="$TRAIN_DATASETS,epickitchens/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
    TRAIN_DATASETS="$TRAIN_DATASETS,egoexolearn/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
    TRAIN_DATASETS="$TRAIN_DATASETS,wtag/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@5"
    TRAIN_DATASETS="$TRAIN_DATASETS,assembly101/${DATA_TYPE}_train_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}@2"
fi


EVAL_DATASETS="sthsthv2/narration_val_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
for DATASET in ego4d holoassist; do
# for DATASET in ego4d holoassist epickitchens egoexolearn wtag assembly101; do
    EVAL_DATASETS="$EVAL_DATASETS,$DATASET/${DATA_TYPE}_val_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
done
EVAL_DATASETS="$EVAL_DATASETS,ego4d/narration_val_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"
EVAL_DATASETS="$EVAL_DATASETS,ego4d/summary_val_L${MAX_SEQ_LEN}_I${NUM_IMG_TOKENS}${SEP_SUFFIX}"

echo $TRAIN_DATASETS
echo $EVAL_DATASETS

# export everything
export MAX_SEQ_LEN
export USE_IMG_CLS_TOKEN
export IMG_PATCH_NUM
export IMG_SEP_TOKEN
export TRAIN_DATASETS
export EVAL_DATASETS
export EPOCHS
export BATCH_SIZE_PER_DEVICE
export GRAD_ACCUM_STEPS
export LR
export NUM_WORKERS
export NFSR
export OUTDIR
export EXP_NAME
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export NUM_GPUS_PER_NODE
export FT_MODULES
export USE_BINARY_HEAD
export BINARY_WEIGHT
export USE_POSE
export BINARY_TYPE
export ZERO_CONFIG

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"

srun --jobid $SLURM_JOBID bash -c 'OMP_NUM_THREADS=1 python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS_PER_NODE \
    --nnodes  $SLURM_JOB_NUM_NODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    mmassist/train/train.py \
    --deepspeed deepspeed/${ZERO_CONFIG}.json \
    --llm_pretrained meta-llama/Meta-Llama-3.1-8B-Instruct \
    --vision_pretrained google/siglip-so400m-patch14-384 \
    --vision_hidden_size 1152 \
    --use_img_cls_token $USE_IMG_CLS_TOKEN \
    --img_patch_token_size $IMG_PATCH_NUM \
    --img_sep_token $IMG_SEP_TOKEN \
    --max_seq_len ${MAX_SEQ_LEN} \
    --train_datasets $TRAIN_DATASETS \
    --eval_datasets $EVAL_DATASETS \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_SIZE_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --gradient_checkpointing True \
    --learning_rate $LR \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --dataloader_num_workers $NUM_WORKERS \
    --dataloader_prefetch_factor 2 \
    --bf16 True \
    --tf32 True \
    --report_to all \
    --output_dir $OUTDIR \
    --run_name $EXP_NAME \
    --neg_frame_sampling_rate $NFSR \
    --use_binary_decision_head $USE_BINARY_HEAD \
    --binary_loss_weight $BINARY_WEIGHT \
    --binary_decision_head_type $BINARY_TYPE \
    --llm_train_mode lora \
    --finetune_modules $FT_MODULES \
    --use_pose $USE_POSE \
    2>&1 | tee -a $OUTDIR/train.log'