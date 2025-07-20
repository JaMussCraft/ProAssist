cd /data/home/imzyc/project/proactive-assist/mmassist/datasets/generate
DATA_ROOT_DIR=<set_this>

# auto-evaluate the generated dialogues
# for DATASET in ego4d holoassist egoexolearn epickitchens wtag assembly101; do
#     for SPLIT in train val; do
#         python auto_eval.py --file ${DATA_ROOT_DIR}/processed_data/$DATASET/generated_dialogs/$SPLIT.json
#     done
# done


# split the generated dialogues into val/test sets
DATA_ROOT_DIR=<set_this>
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/ego4d --train_filter_score 3 --sampling_ratio 1 --eval_filter_score 5
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/holoassist --train_filter_score 3 --sampling_ratio 0.5 --eval_filter_score 5
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/egoexolearn --train_filter_score 3 --sampling_ratio 1 --eval_filter_score 5
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/epickitchens --train_filter_score 3 --sampling_ratio 2 --eval_filter_score 5
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/wtag --train_filter_score 3 --sampling_ratio 2 --eval_filter_score 5
python filter_and_split.py --data_dir ${DATA_ROOT_DIR}/processed_data/assembly101 --train_filter_score 3 --sampling_ratio 1 --eval_filter_score 5

# python split_data.py --data_dir /fsx_0/user/imzyc/processed_data/wtag --filter_score 5