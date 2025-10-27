# SmolVLM2 Fine-tuning on ProAssist Dataset

Quick guide for fine-tuning SmolVLM2 models on the ProAssist dataset using LoRA adapters.

## Environment Setup

Create the conda environment from the provided YAML file:

```bash
conda env create -f smolvlm_environment.yml
conda activate smolvlm
```

## Quick Start

### Basic Usage

```bash
python finetune_smolvlm.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
    --use_lora true \
    --data_root_dir /path/to/proassist_data/processed_data/ \
    --train_datasets wtag/dialog-klg-sum_train_L2048_I1 \
    --eval_datasets wtag/dialog-klg-sum_val_L2048_I1 \
    --output_dir ./my_adapter \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

### Key Parameters

**Model Configuration:**
- `--model_name_or_path`: Base model (`HuggingFaceTB/SmolVLM2-2.2B-Instruct` or `SmolVLM2-500M-Instruct`)
- `--use_lora`: Enable LoRA training (saves only adapter weights, recommended)
- `--use_qlora`: Enable 4-bit quantization for reduced memory
- `--freeze_vision`: Freeze vision encoder (default: false)

**Dataset Configuration:**
- `--data_root_dir`: Root directory containing processed ProAssist data
- `--train_datasets`: Training dataset name(s)
- `--eval_datasets`: Evaluation dataset name(s)
- `--frame_sampling_ratio`: Ratio of frames to sample from ranges (default: 0.1)
- `--w2t_frame_sampling_rate`: Sampling rate for negative frames (default: 0.3)

**Training Configuration:**
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per GPU (default: 2)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--output_dir`: Directory to save the adapter

**Loss Configuration:**
- `--use_inverse_freq_weighting`: Use inverse frequency weighting for balanced learning
- `--use_end_of_utterance_for_w2t`: Use end-of-utterance token for speaking decisions
- `--w2t_only`: Only compute loss for speaking decision tokens
- `--no_assistant`: Disable learning from assistant response tokens

**LoRA Configuration:**
- `--lora_r`: LoRA rank (default: 16, increase if underfitting)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1, increase if overfitting)

## Training Output

After training, the output directory contains:
```
./my_adapter/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # Adapter weights
├── adapter_metadata.json         # Training metadata
├── tokenizer files...
├── processor files...
└── checkpoint-X/                 # Checkpoints at save_steps intervals
```

## Using Trained Adapters

### 1. Single Adapter Inference

```python
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import torch

# Load base model
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./my_adapter")
processor = AutoProcessor.from_pretrained("./my_adapter")

# Run inference
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "What's happening?"}
    ]
}]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

### 2. Combining Multiple Adapters

Train separate adapters on different datasets:

```bash
# Train adapter 1
python finetune_smolvlm.py \
    --train_datasets dataset1 \
    --output_dir ./adapter1 \
    --num_train_epochs 2

# Train adapter 2  
python finetune_smolvlm.py \
    --train_datasets dataset2 \
    --output_dir ./adapter2 \
    --num_train_epochs 2
```

**Combine with weighted averaging:**

```bash
python combine_lora_adapters.py \
    --adapters ./adapter1 ./adapter2 \
    --weights 0.7 0.3 \
    --output ./merged_adapter
```

**Create multi-adapter model (switchable):**

```bash
python combine_lora_adapters.py \
    --adapters ./adapter1 ./adapter2 \
    --mode multi \
    --output ./multi_adapter
```

### 3. Multi-Adapter Usage

```python
from peft import PeftModel

# Load base model
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load multiple adapters
model = PeftModel.from_pretrained(base_model, "./adapter1", adapter_name="adapter1")
model.load_adapter("./adapter2", adapter_name="adapter2")

# Switch between adapters
model.set_adapter("adapter1")  # Use adapter1
# ... run inference ...

model.set_adapter("adapter2")  # Switch to adapter2
# ... run inference ...
```

## Example SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=finetune_smolvlm
#SBATCH --account=<account_name>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          
#SBATCH --cpus-per-task=8      
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=./finetune_smolvlm/finetune_smolvlm_%j.out
#SBATCH --error=./finetune_smolvlm/finetune_smolvlm_%j.err

source ~/.bashrc
conda activate smolvlm
export PYTHONPATH=/path/to/ProAssist:$PYTHONPATH

python finetune_smolvlm.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
    --use_lora true \
    --data_root_dir /path/to/processed_data/ \
    --train_datasets wtag/dialog-klg-sum_train_L2048_I1 \
    --eval_datasets wtag/dialog-klg-sum_val_L2048_I1 \
    --output_dir ./my_adapter \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --use_inverse_freq_weighting true \
    --use_end_of_utterance_for_w2t true
```

## Evaluating Speaking Decisions (Eval-Only Mode)

Run speaking decision evaluation on a checkpoint without training:

**Evaluate a trained checkpoint:**
```bash
python finetune_smolvlm.py \
    --speaking_eval_only \
    --speaking_eval_checkpoint ./my_adapter/checkpoint-1000 \
    --use_lora \
    --eval_datasets wtag/dialog-klg-sum_val_L2048_I1 \
    --data_root_dir /path/to/processed_data/ \
    --output_dir ./eval_results_checkpoint \
    --per_device_eval_batch_size 8
```

## Tips

1. **Monitoring training:** Use TensorBoard with `tensorboard --logdir ./my_adapter/runs`
2. **Evaluate checkpoints:** Use `--speaking_eval_only` to compare checkpoint performance


## See Also

- `combine_lora_adapters.py` - Adapter combination utility
- `finetune_jobs/` - Example SLURM scripts
