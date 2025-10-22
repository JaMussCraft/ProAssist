# LoRA Adapter Training and Combination Guide

This guide explains how to train separate LoRA adapters on different datasets and combine them for improved model performance.

## Overview

The fine-tuning script (`finetune_smolvlm.py`) has been configured to save LoRA adapters in a format that's easy to combine later. Each adapter is saved with:

- **adapter_config.json**: LoRA configuration
- **adapter_model.safetensors** (or .bin): Adapter weights only (not full model)
- **adapter_metadata.json**: Training information (dataset, hyperparameters, etc.)

## Training Separate Adapters

### Step 1: Train on Each Dataset

Train a separate adapter for each dataset by launching the script with different parameters:

```bash
# Adapter 1: Train on dataset A
python finetune_smolvlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --use_lora True \
    --train_datasets wtag/dialog-klg-sum_train_L2048_I1 \
    --output_dir ./adapters/adapter_dataset_A \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8

# Adapter 2: Train on dataset B
python finetune_smolvlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --use_lora True \
    --train_datasets another_dataset_name \
    --output_dir ./adapters/adapter_dataset_B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8

# Adapter 3: Train on dataset C
python finetune_smolvlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --use_lora True \
    --train_datasets yet_another_dataset \
    --output_dir ./adapters/adapter_dataset_C \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8
```

**Important**: Always use the same:
- Base model (`--model_name_or_path`)
- LoRA configuration (r, alpha, dropout, target_modules)

This ensures adapters are compatible for combination.

### Step 2: Verify Saved Adapters

After training, each adapter directory should contain:

```
./adapters/adapter_dataset_A/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # Adapter weights (small, ~10-100MB)
├── adapter_metadata.json         # Training info
├── preprocessor_config.json      # Processor configs
├── tokenizer files...
└── training_args.bin
```

Check the metadata:

```bash
cat ./adapters/adapter_dataset_A/adapter_metadata.json
```

## Combining Adapters

### Method 1: Weighted Average Combination (Recommended)

Combine adapters using weighted averaging to create a single merged adapter:

```bash
# Equal weighting (default)
python combine_lora_adapters.py \
    --adapters ./adapters/adapter_dataset_A ./adapters/adapter_dataset_B \
    --output ./adapters/combined_equal \
    --mode weighted

# Custom weighting (weights must sum to 1.0)
python combine_lora_adapters.py \
    --adapters ./adapters/adapter_dataset_A ./adapters/adapter_dataset_B ./adapters/adapter_dataset_C \
    --weights 0.5 0.3 0.2 \
    --output ./adapters/combined_weighted \
    --mode weighted
```

**When to use**: When you want a single adapter that blends knowledge from multiple datasets.

### Method 2: Multi-Adapter Model (Runtime Switching)

Create a model with multiple adapters that can be switched at runtime:

```bash
python combine_lora_adapters.py \
    --adapters ./adapters/adapter_dataset_A ./adapters/adapter_dataset_B \
    --output ./adapters/multi_adapter \
    --mode multi
```

**When to use**: When you want to dynamically select which adapter to use during inference.

## Using Combined Adapters

### Using Weighted Average Adapter

```python
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Load base model
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load combined adapter
model = PeftModel.from_pretrained(base_model, "./adapters/combined_weighted")
processor = AutoProcessor.from_pretrained("./adapters/combined_weighted")

# Use for inference
inputs = processor(text="...", images=[...], return_tensors="pt")
outputs = model.generate(**inputs)
```

### Using Multi-Adapter Model

```python
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Load base model
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load multi-adapter model
model = PeftModel.from_pretrained(base_model, "./adapters/multi_adapter")

# Switch between adapters
model.set_adapter("adapter_0")  # Use first adapter
# ... run inference ...

model.set_adapter("adapter_1")  # Switch to second adapter
# ... run inference ...
```

### Merging Adapter into Base Model

To create a standalone model without LoRA (for faster inference):

```python
from peft import PeftModel
from transformers import Idefics3ForConditionalGeneration

base_model = Idefics3ForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./adapters/combined_weighted")

# Merge and unload (creates standalone model)
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./models/merged_smolvlm")
```

## Best Practices

### 1. Consistent Training Configuration

Always use the same LoRA configuration across all adapters:

```python
# Example consistent configuration
--lora_r 16
--lora_alpha 32
--lora_dropout 0.1
--lora_target_modules "down_proj,o_proj,k_proj,q_proj,gate_proj,up_proj,v_proj"
```

### 2. Adapter Weighting Strategy

Choose weights based on:
- **Dataset size**: Larger datasets → higher weight
- **Dataset quality**: Higher quality → higher weight
- **Task importance**: More critical tasks → higher weight
- **Performance**: Better-performing adapters → higher weight

Example:
```bash
# Dataset A: 10k samples, high quality
# Dataset B: 5k samples, medium quality  
# Dataset C: 2k samples, low quality
# Weights: roughly proportional to size × quality

python combine_lora_adapters.py \
    --adapters ./adapter_A ./adapter_B ./adapter_C \
    --weights 0.6 0.3 0.1 \
    --output ./combined
```

### 3. Validation

After combining adapters, validate on a held-out test set:

```python
# Test combined adapter
from mmassist.train.finetune_smolvlm import SmolVLMProAssistTrainer

trainer = SmolVLMProAssistTrainer(...)
metrics = trainer.evaluate_speaking_decisions(eval_dataset)
print(metrics)
```

### 4. Version Control

Keep track of adapter combinations:

```bash
# Save combination info
echo "adapter_A (0.5) + adapter_B (0.3) + adapter_C (0.2)" > ./combined/combination_info.txt

# Or check the metadata
cat ./combined/combined_adapter_metadata.json
```

## Troubleshooting

### Issue: Incompatible adapter configurations

**Error**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`

**Solution**: Ensure all adapters use the same:
- Base model
- LoRA rank (r)
- Target modules
- LoRA alpha

### Issue: Combined adapter performs worse than individual adapters

**Possible causes**:
1. **Dataset conflict**: Datasets teach contradictory behaviors
2. **Poor weighting**: Try adjusting weights
3. **Overfitting**: Individual adapters may be overfitted to their specific datasets

**Solutions**:
- Use multi-adapter mode and switch based on input type
- Experiment with different weight combinations
- Add regularization during training

### Issue: Out of memory when combining

**Solution**: Combine adapters on CPU or use gradient checkpointing:

```bash
# Force CPU for combination
CUDA_VISIBLE_DEVICES="" python combine_lora_adapters.py --adapters ...
```

## Advanced: Manual Weighted Combination

For fine-grained control, manually combine adapter weights:

```python
import torch
from collections import OrderedDict

# Load adapter state dicts
adapter1_state = torch.load("adapter1/adapter_model.bin")
adapter2_state = torch.load("adapter2/adapter_model.bin")

# Weighted average
combined_state = OrderedDict()
weight1, weight2 = 0.7, 0.3

for key in adapter1_state.keys():
    combined_state[key] = weight1 * adapter1_state[key] + weight2 * adapter2_state[key]

# Save combined state
torch.save(combined_state, "combined/adapter_model.bin")
```

## Summary

1. **Train** separate adapters with `finetune_smolvlm.py` using `--use_lora True`
2. **Combine** adapters with `combine_lora_adapters.py`
3. **Use** combined adapter for inference
4. **Validate** on test sets to ensure improved performance

For questions or issues, check the adapter metadata files or training logs.
