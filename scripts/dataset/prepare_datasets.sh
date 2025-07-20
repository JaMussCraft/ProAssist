cd mmassist/datasets/generate
SEP="EOS"

################## Prepare the Stage-1 Vision-Language Alignment Data ##################
# for MAX_SEQ_LEN in 2048 4096 8192; do
for MAX_SEQ_LEN in 4096; do
    # I=1, 5, 10
    for IMG_PATCH_TOKEN_SIZE in 0 2 3; do
        # prepare ego4d narrations
        python prepare_ego4d_narrations.py --max_seq_len $MAX_SEQ_LEN --img_patch_token_size $IMG_PATCH_TOKEN_SIZE --img_sep_token $SEP

        # prepare sthsth
        python prepare_sthsth.py --split train --max_seq_len $MAX_SEQ_LEN --img_patch_token_size $IMG_PATCH_TOKEN_SIZE --img_sep_token $SEP
        python prepare_sthsth.py --split val --max_seq_len $MAX_SEQ_LEN --img_patch_token_size $IMG_PATCH_TOKEN_SIZE --img_sep_token $SEP

        # prepare llava
        python prepare_llava.py --max_seq_len $MAX_SEQ_LEN --img_patch_token_size $IMG_PATCH_TOKEN_SIZE --img_sep_token $SEP

        # prepare egoobjects
        python prepare_egoobj.py --max_seq_len $MAX_SEQ_LEN --img_patch_token_size $IMG_PATCH_TOKEN_SIZE --img_sep_token $SEP
    done
done


################## Prepare the Stage-2 Dialog Data ##################
for DATASET in ego4d holoassist egoexolearn epickitchens wtag assembly101; do
    # I=1, 5, 10
    for IMG_PATCH_TOKEN_SIZE in 0 2 3; do 
        
        # prepare the dialog data with a cut-off length
        # for MAX_SEQ_LEN in 2048 4096 8192; do
        for MAX_SEQ_LEN in 4096; do
            # training data and validation data for computing eval_loss
            echo "DATASET: $DATASET, IMG_PATCH_TOKEN_SIZE: $IMG_PATCH_TOKEN_SIZE, MAX_SEQ_LEN: $MAX_SEQ_LEN"
            for SPLIT in train val test; do
                for ADD_KNOWLEDGE in True False; do
                    python prepare_dialogs.py \
                        --dataset $DATASET \
                        --split $SPLIT \
                        --max_seq_len $MAX_SEQ_LEN \
                        --img_patch_token_size $IMG_PATCH_TOKEN_SIZE \
                        --add_knowledge $ADD_KNOWLEDGE \
                        --img_sep_token $SEP
                done
            done
            
            # summary only data
            SUMMARY_LEN=$((MAX_SEQ_LEN / 4))
            for SPLIT in train val; do
                python prepare_dialogs.py \
                    --dataset $DATASET \
                    --split $SPLIT \
                    --max_seq_len $MAX_SEQ_LEN \
                    --img_patch_token_size $IMG_PATCH_TOKEN_SIZE \
                    --summary_only True \
                    --reserved_max_summary_len $SUMMARY_LEN \
                    --img_sep_token $SEP
            done
        done
        
        # prepare the full-length dialog data for inference
        for SPLIT in val test; do
            for ADD_KNOWLEDGE in True False; do
                python prepare_dialogs.py \
                    --dataset $DATASET \
                    --split $SPLIT \
                    --max_seq_len 0 \
                    --img_patch_token_size $IMG_PATCH_TOKEN_SIZE \
                    --add_knowledge $ADD_KNOWLEDGE \
                    --add_summary False \
                    --img_sep_token $SEP
            done
        done
    done
done