## Proactive Assistant Dialogue Generation from Streaming Egocentric Videos
Corresponding author: [Yichi Zhang](https://594zyc.github.io/)

[Paper](https://arxiv.org/abs/2506.05904) | [Project Page](https://pro-assist.github.io/) | [Dataset](https://huggingface.co/594zyc/ProAssist-Dataset) | [Model](https://huggingface.co/collections/594zyc/proassist-687abdb5957fd79512465c68) 


### Installation
1. Clone the repo

```bash
git clone https://github.com/pro-assist/ProAssist.git
cd ProAssist
```

2. (Optional) Create a virtual environment

```shell
conda create -n mm python=3.10 -y
conda activate mm
```

3. Install dependencies

```shell
pip install -r requirements.txt
pip install -e .
```


### Data Preparation 

1. Set the data root dir in `mmassist/configs/arguments.py`, or export `DATA_ROOT_DIR` in your environment.
```
export DATA_ROOT_DIR=<your_data_root_dir>
```

2. Download the preprocessed data:
```
git lfs install
git clone https://huggingface.co/594zyc/ProAssist-Dataset
mv ProAssist-Dataset/processed_data $DATA_ROOT_DIR/processed_data
```

Note: the preprocessed data is 152 GB with many files, so it is slow to download. To download a subset of the data for preview, you can use the following command:
```
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/594zyc/ProAssist-Dataset
git lfs pull -I "processed_data/wtag"  # will only download the wtag subset
```

3. Unzip the data:
```
for dataset in ego4d holoassist epickitchens egoexolearn wtag assembly101; do
    cd $DATA_ROOT_DIR/processed_data/$dataset
    unzip generated_dialogs.zip
    unzip prepared.zip
done
```

If you want to prepare the data from scratch using the LLM-based data generation pipeline, please see [here](mmassist/datasets/README.md).


### Model Download
```
cd $DATA_ROOT_DIR
mkdir -p models && cd models

# download the I=1 model (1 token per frame)
git clone https://huggingface.co/594zyc/ProAssist-Model-L4096-I1

# download the I=5 model (5 tokens per frame)
git clone https://huggingface.co/594zyc/ProAssist-Model-L4096-I5

# download the I=10 model (10 tokens per frame)
git clone https://huggingface.co/594zyc/ProAssist-Model-L4096-I10
```

### Playground
We provide several notebooks to demonstrate:
1. Video and dialogue visualization ([link](notebooks/video_visualization.ipynb))
2. Model inference for streaming video-to-dialogue generation ([link](notebooks/streaming_inference.ipynb))
3. LLM-based dialogue generation pipeline ([link](notebooks/dialog_generation_playground.ipynb))
4. LLM-as-a-judge evaluation ([link](notebooks/llm_eval_playground.ipynb))
5. Dataset statistics overview ([link](notebooks/data_analysis.ipynb))

### Training & Evaluation
Note: the training and evaluation scripts only work with the slurm cluster currently.
```
# Train the I=1, 5, 10 model (I=#tokens/frame) 
sbatch scripts/train/I1_8n_4096_1s.sh
sbatch scripts/train/I5_12n_4096_1s.sh
sbatch scripts/train/I10_16n_4096_1s.sh

# Evaluate a trained model
sbatch scripts/eval/Aug_eval_stream.sh
```

## Citation <a name="citation"></a>
Please consider citing our paper if you find this project helpful for your research:

```bibtex
@article{zhang2025proactive,
  title={Proactive Assistant Dialogue Generation from Streaming Egocentric Videos},
  author={Zhang, Yichi and Dong, Xin Luna and Lin, Zhaojiang and Madotto, Andrea and Kumar, Anuj and Damavandi, Babak and Chai, Joyce and Moon, Seungwhan},
  journal={arXiv preprint arXiv:2506.05904},
  year={2025}
}
```