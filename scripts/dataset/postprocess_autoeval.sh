cd mmassist/datasets/generate

# auto-evaluate the generated dialogues
for DATASET in ego4d holoassist egoexolearn epickitchens wtag assembly101; do
    for SPLIT in train val; do
        python auto_eval.py --file ${DATA_ROOT_DIR}/processed_data/$DATASET/generated_dialogs/$SPLIT.json
    done
done