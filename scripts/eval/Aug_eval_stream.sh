L=4096
I=1

# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240816-L${L}-I${I}-ep4-NOSEP-nr1.0-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240816-L${L}-I${I}-ep4-NOSEP-nr0.2-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240818-L${L}-I${I}-ep4-NOSEP-nr1.0-w2t_head_w0.2-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240818-L${L}-I${I}-ep4-NOSEP-nr1.0-w2t_head_w0.5-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240815-L${L}-I${I}-ep4-NOSEP-nr0.2-s2-lora
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240816-L4096-I1-ep8-NOSEP-nr0.2-1s-lora-bs256

# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240816-L4096-I10-ep8-NOSEP-nr0.2-1s-lora-bs32
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240817-L4096-I1-ep4-NOSEP-nr0.2-w2t_head-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240820-L4096-I1-ep4-NOSEP-nr0.2-klgmix-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240820-L4096-I1-ep4-NOSEP-nr0.1-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240820-L4096-I1-ep4-NOSEP-nr0.01-1s-lora-bs256

# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240820-L4096-I1-ep4-NOSEP-nr0.1-w2t_head_w0.2-1s-lora-bs256
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256


### models not trained yet
# MODEL_NAME=20240822-L4096-I26-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs1024
# RATES="0.1, 0.2, 0.3 0.4"



# I=10
# L=4096
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512

# SETUPS="ego4d/narration_val_L${L}_I${I}|offline|4k|0.5|none"
# SETUPS="sthsthv2/narration_val_L${L}_I${I}|offline|4k|0.5|none"
# SETUPS="ego4d/narration_val_L131072_I${I}|offline|32k|0.7|none"
# SETUPS="$SETUPS,ego4d/narration_val_L131072_I${I}|offline|32k|0.8|none"
# for DATASET in ego4d holoassist epickitchens egoexolearn wtag assembly101; do
#     # SETUPS="$SETUPS,$DATASET/dialog_val_L0_I${I}|offline|4k|0.5|none"
#     SETUPS="$SETUPS,$DATASET/dialog_val_L0_I${I}|offline|32k|0.5|none"
#     SETUPS="$SETUPS,$DATASET/dialog-sum_val_L${L}_I${I}|offline|4k|0.5|none"
# done



RUNNER="stream"
CTX_METHOD="summarize_and_drop"

# SETUPS="ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.2|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.3|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.4|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.5|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.6|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.7|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.8|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.9|$CTX_METHOD"

# SETUPS="ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.2|$CTX_METHOD"
# SETUPS="$SETUPS,ego4d/narration_val_L${L}_I${I}|$RUNNER|4k|0.3|$CTX_METHOD"


##########################################################################################

# DIALOG EXPERIMENTS TO RUN

# THRESHOLD SELECTION

ADD_KNOWLEDGE=""
SPLIT=val

# JOBID: 13821 (CG) -> 13831
# MODEL_NAME=20240816-L4096-I1-ep4-NOSEP-nr1.0-1s-lora-bs256
# RATES="0.5 0.6 0.7 0.8"

# JOBID: 13820 (CG) -> 13838
# MODEL_NAME=20240816-L4096-I1-ep4-NOSEP-nr0.2-1s-lora-bs256
# RATES="0.1 0.2 0.3 0.4"

# JOBID: 13819 (CG) -> 13837
# MODEL_NAME=20240820-L4096-I1-ep4-NOSEP-nr0.1-1s-lora-bs256
# RATES="0.1 0.2 0.3 0.4"

# JOBID: 13818 (CG) -> 13835
# MODEL_NAME=20240820-L4096-I1-ep4-NOSEP-nr0.1-1s-lora-bs256
# RATES="0.1 0.2 0.3"

# JOBID: 13816 (CG) -> 13834
# MODEL_NAME=20240820-L4096-I1-ep4-NOSEP-nr0.01-1s-lora-bs256
# RATES="0.1 0.2 0.3"


# JOBID: 13814 (CG) -> 13833 (CG) -> 13836 -> 13994
# MODEL_NAME=20240820-L4096-I1-ep4-NOSEP-nr0.1-w2t_head_w0.2-1s-lora-bs256
# RATES="0.1 0.2 0.3 0.4"


# JOBID: 14225
# MODEL_NAME=20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256
# RATES="0.1 0.2 0.3 0.4 0.5"


# JOBID: 13813 (CG) -> 13832 -> 13998
# I=10
# MODEL_NAME=20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512
# RATES="0.5"
# RATES="0.1 0.2 0.3 0.4"


# JOBID: 14217
# MODEL_NAME=20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-pose-1s-lora-bs512
# I=10
# RATES="0.1 0.2 0.3 0.4 0.5"

# JOBID: 14222
# I=5
# MODEL_NAME=20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug
# RATES="0.1 0.2 0.3 0.4 0.5"





# SINGLE BEST of I=1, streaming and offline (x2)
# 1. CTX_METHOD=drop_middle




# SINGLE BEST of I=1, 5, 10, 26, STREAMING AND OFFLINE (4x2)
# 1. ON TEST



# 2.1 w/ KNOWLEDGE VAL and TEST

SPLIT=val
ADD_KNOWLEDGE="-klg"

## 14226 14290 r
# I=1
# MODEL_NAME=20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256
# RATES="0.2 0.3 0.4 0.5"

## 14291
I=5
MODEL_NAME=20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug
RATES="0.2 0.3 0.4 0.5"

##
# I=10
# MODEL_NAME=20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512
# RATES="0.2 0.3 0.4 0.5"

# 2.2 w/ KNOWLEDGE TEST




# LLM evaluation on
# 1. SINGLE BEST rate of I=1, 5, 10, 26
# 2. add knowledge or not (x2)
# 3. val and test (x2)

MODEL_PATH=/fsx_0/user/imzyc/proact_exps/$MODEL_NAME

SETUPS="ego4d/dialog_val_L0_I${I}|$RUNNER|4k|0.05|$CTX_METHOD"
for RATE in $RATES; do
    for DATASET in ego4d holoassist epickitchens egoexolearn wtag assembly101; do
        SETUPS="$SETUPS,$DATASET/dialog${ADD_KNOWLEDGE}_${SPLIT}_L0_I${I}|$RUNNER|4k|$RATE|$CTX_METHOD"
    done
done


echo $MODEL_PATH
echo $SETUPS

OMP_NUM_THREADS=4 python mmassist/eval/eval.py \
    --model_path $MODEL_PATH \
    --inference_setups $SETUPS \
    --job_name eval \
    --num_nodes 8 \
    --force_rerun False \
    --slurm_exclude h100-st-p548xlarge-[150,152-153,164] \
    2>&1 | tee -a $MODEL_PATH/eval.log

# --account ar-ai-research-interns \
n