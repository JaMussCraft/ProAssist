RAW_DATA_DIR=${DATA_ROOT_DIR}/datasets
PROCESSED_DATA_DIR=${DATA_ROOT_DIR}/processed_data

# Ego4D
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/ego4d/frames --output_dir $PROCESSED_DATA_DIR/ego4d/features

# Holoassist
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/holoassist/frames --output_dir $PROCESSED_DATA_DIR/holoassist/features

# Epic-Kitchens
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/epickitchens/frames --output_dir $PROCESSED_DATA_DIR/epickitchens/features

# EgoExoLearn
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/egoexolearn/frames --output_dir $PROCESSED_DATA_DIR/egoexolearn/features

# WTaG
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/wtag/frames --output_dir $PROCESSED_DATA_DIR/wtag/features

# Assembly101
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/assembly101/frames --output_dir $PROCESSED_DATA_DIR/assembly101/features

# LLaVA Images
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/llava/frames --output_dir $PROCESSED_DATA_DIR/llava/features

# SomethingSomethingV2
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/sthsthv2/frames --output_dir $PROCESSED_DATA_DIR/sthsthv2/features

# EgoObjects
python mmassist/datasets/preprocess/encode_frames.py --preprocessed_frames_dir $PROCESSED_DATA_DIR/egoobjects/frames --output_dir $PROCESSED_DATA_DIR/egoobjects/features