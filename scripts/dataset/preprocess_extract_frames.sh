RAW_DATA_DIR=${DATA_ROOT_DIR}/datasets
PROCESSED_DATA_DIR=${DATA_ROOT_DIR}/processed_data

# Ego4D
python mmassist/datasets/preprocess/extract_frames.py --dataset ego4d --video_dir $RAW_DATA_DIR/ego4d_track2/v2/video_540ss --output_dir $PROCESSED_DATA_DIR/ego4d/frames

## Ego4D narrations
python mmassist/datasets/preprocess/preprocess_ego4d_narrations.py

# Holoassist
python mmassist/datasets/preprocess/extract_frames.py --dataset holoassist --video_dir $RAW_DATA_DIR/holoassist/video_pitch_shifted --output_dir $PROCESSED_DATA_DIR/holoassist/frames

# Epic-Kitchens
python mmassist/datasets/preprocess/extract_frames.py --dataset epickitchens --video_dir $RAW_DATA_DIR/epic-kitchens --output_dir $PROCESSED_DATA_DIR/epickitchens/frames --num_nodes 4

# EgoExoLearn
python mmassist/datasets/preprocess/extract_frames.py --dataset egoexolearn --video_dir $RAW_DATA_DIR/EgoExoLearn/videos --output_dir $PROCESSED_DATA_DIR/egoexolearn/frames

# WTaG
python mmassist/datasets/preprocess/extract_frames.py --dataset wtag --video_dir $RAW_DATA_DIR/WTaG --output_dir $PROCESSED_DATA_DIR/wtag/frames

# Assembly101
python mmassist/datasets/preprocess/extract_frames.py --dataset assembly101 --video_dir $RAW_DATA_DIR/assembly101/videos --output_dir $PROCESSED_DATA_DIR/assembly101/frames

# EPFL
python mmassist/datasets/preprocess/extract_frames.py --dataset epfl --video_dir $RAW_DATA_DIR/epfl/videos --output_dir $PROCESSED_DATA_DIR/epfl/frames

# LLaVA Image Captioning
python mmassist/datasets/preprocess/preprocess_llava_images.py --ann_file $RAW_DATA_DIR/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --image_root $RAW_DATA_DIR/LLaVA-Pretrain/images --output_dir $PROCESSED_DATA_DIR/llava

# SomethingSomethingV2
python mmassist/datasets/preprocess/preprocess_sthsthv2.py

# EgoObjects
python mmassist/datasets/preprocess/preprocess_egoobjects.py