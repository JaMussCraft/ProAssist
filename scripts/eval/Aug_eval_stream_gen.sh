
# MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-pose-1s-lora-bs512
# SETUPS="ego4d/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop"
# SETUPS="$SETUPS,epickitchens/dialog_val_L0_I10|stream|32k|0.2|summarize_and_drop"
# SETUPS="$SETUPS,holoassist/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop"
# SETUPS="$SETUPS,egoexolearn/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop"
# SETUPS="$SETUPS,assembly101/dialog_val_L0_I10|stream|32k|0.3|summarize_and_drop"
# SETUPS="$SETUPS,wtag/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop"


# SETUPS="ego4d/dialog_val_L0_I10|stream|32k|0.4|drop_middle"
# SETUPS="$SETUPS,epickitchens/dialog_val_L0_I10|stream|32k|0.2|drop_middle"
# SETUPS="$SETUPS,holoassist/dialog_val_L0_I10|stream|32k|0.4|drop_middle"
# SETUPS="$SETUPS,egoexolearn/dialog_val_L0_I10|stream|32k|0.4|drop_middle"
# SETUPS="$SETUPS,assembly101/dialog_val_L0_I10|stream|32k|0.3|drop_middle"
# SETUPS="$SETUPS,wtag/dialog_val_L0_I10|stream|32k|0.4|drop_middle"

MODEL_PATH=/fsx_0/user/imzyc/proact_exps/20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug
SETUPS="ego4d/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop"
SETUPS="$SETUPS,epickitchens/dialog-klg_test_L0_I5|stream|4k|0.2|summarize_and_drop"
SETUPS="$SETUPS,holoassist/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop"
SETUPS="$SETUPS,egoexolearn/dialog-klg_test_L0_I5|stream|4k|0.4|summarize_and_drop"
SETUPS="$SETUPS,assembly101/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop"
SETUPS="$SETUPS,wtag/dialog-klg_test_L0_I5|stream|4k|0.5|summarize_and_drop"


echo $MODEL_PATH
echo $SETUPS

OMP_NUM_THREADS=4 python mmassist/eval/eval.py \
    --model_path $MODEL_PATH \
    --inference_setups $SETUPS \
    --job_name eval \
    --num_nodes 8 \
    --force_rerun False \
    2>&1 | tee -a $MODEL_PATH/eval.log

# --account ar-ai-research-interns \