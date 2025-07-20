

## Set the data root dir
```
export DATA_ROOT_DIR=<your_data_root_dir>
```
or replace `<please_set_me>` in `mmassist/configs/arguments.py`.

## Download the Raw Videos and Annotations
We use the videos and annotations from 6 sources:
* [Ego4D](https://ego4d-data.org/)
* [EpicKitchen](https://epic-kitchens.github.io/2024)
* [HoloAssist](https://holoassist.github.io/)
* [Assembly101](https://assembly-101.github.io/)
* [EgoExoLearn](https://github.com/OpenGVLab/EgoExoLearn)
* [WTaG](https://github.com/sled-group/Watch-Talk-and-Guide).

In addition to them, we also use some auxiliary data for training:
* [LLaVA-Pretrain-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
* [SomethingSomething-V2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [EgoObjects](https://github.com/facebookresearch/EgoObjects?tab=readme-ov-file)


Please download all the data follow the instruction on their publisher and organize them into the following structure:

```
<DATA_ROOT_DIR>/datasets
├── EgoExoLearn
│   ├── LICENSE
│   ├── README.md
│   ├── action_anticipation_planning_benchmark
│   ├── annotations
│   ├── ...
│   ├── videos
│   └── videos.zip
├── EgoObjects
│   ├── EgoObjectsV1_unified_train.json
│   ├── images
│   └── images.zip
├── LLaVA-Pretrain
│   ├── README.md
│   ├── blip_laion_cc_sbu_558k.json
│   ├── blip_laion_cc_sbu_558k_meta.json
│   ├── images
│   └── images.zip
├── WTaG
│   ├── T1
│   ├── T2
│   ├── ...
│   └── T55b
├── assembly101
│   ├── annotations
│   ├── assembly101-download-scripts
│   └── videos
├── captaincook4d
│   ├── annotations
│   ├── captain_cook_4d
│   └── downloader
├── ego4d_track2
│   ├── ego4d.json
│   └── v2
├── epic-kitchens
│   ├── EK100
│   ├── EK55
│   ├── download_utils
│   ├── epic-kitchens-100-annotations
│   └── epic-kitchens-download-scripts
├── holoassist
│   ├── data-annotation-trainval-v1_1.json
│   ├── data-splits-v1_2.zip
│   ├── test-v1_2.txt
│   ├── train-v1_2.txt
│   ├── val-v1_2.txt
│   ├── video_pitch_shifted
│   └── video_pitch_shifted.tar
└── sthsth-v2
    ├── 20bn-something-something-download-package-labels.zip
    ├── OpenDataLab___sthv2
    ├── download.py
    ├── labels
    └── videos
```

## Data Generation and Preparation Instructions

Important note: the following scripts are tested with SLURM environment where each node has 8 80G H100 GPUs. We use vLLM to serve the 70B LLaMA-3.1-70B-Instruct model, and split it into 4 GPUs per node. Reproducing the whole data generation process can be costly.

To run the data preparation process from scratch, please follow the following steps:

1. Extract frames from videos at 2 FPS, resize & crop them to 384x384, and save them in the `${DATA_ROOT_DIR}/processed_data/{subset_name}/frames` directories.  
```
# Note: it is recommended to run this script in a SLURM environment.
sh scripts/dataset/preprocess_extract_frames.sh
```

2. Dialogue generation using LLM. We use LLaMA-3.1-70B-Instruct to generate the dialogs from available annotations of each dataset, including goal description, step descriptions, task knowledge, and existing dialogs (if applicable). 
```
sh scripts/dataset/generate_dialog.sh
```

3. Auto-evaluate the generated dialogues using LLM-as-a-judge.
```
sh scripts/dataset/postprocess_autoeval.sh
```

4. Split the generated dialogues into train/val/test sets.
```
sh scripts/dataset/postprocess_filter_split.sh
```

Steps below are only for model training:

5. Encode each frame using the `siglip-so400m-patch14-384` model and save the features. 
```
sh scripts/dataset/preprocess_encode_frames.sh
```

6. Training sample preparation: split long videos into training samples of a maximum sequence length (e.g. 4096), under different experimental setups (e.g. with knowledge or not).
```
sh scripts/dataset/prepare_datasets.sh
```