# SmolVLM ProAssist Fine-tuning Pipeline

This directory contains a complete pipeline for fine-tuning SmolVLM-Instruct on the ProAssist dataset for streaming video assistance tasks.

## Overview

The pipeline converts ProAssist's multi-modal conversation format (with temporal video frames) to SmolVLM's chat template format and implements efficient training with LoRA/QLoRA support.

## Key Features

- **ProAssist Dataset Integration**: Seamlessly converts ProAssist conversations to SmolVLM format
- **Temporal Frame Processing**: Handles video frames with temporal context from ProAssist
- **Memory Efficient Training**: Supports LoRA and QLoRA for reduced memory usage
- **4:1 Aspect Ratio Optimization**: Optional image resizing for optimal SmolVLM encoding
- **Configurable Frame Sampling**: Control number of frames per training sample
- **Vision Encoder Control**: Option to freeze/unfreeze vision components

## Files

### Core Training Files
- `finetune_smolvlm.py`: Main training script with ProAssist dataset integration
- `train_smolvlm_proassist.sh`: Training script with optimized parameters
- `test_smolvlm_training.py`: Test script to validate setup

### Configuration
- `../configs/smolvlm_proassist_config.yaml`: Configuration template
- This README file

## Quick Start

### 1. Test the Setup
```bash
cd /u/swong2/ProAssist
python test_smolvlm_training.py
```

### 2. Run Training
```bash
chmod +x scripts/train/train_smolvlm_proassist.sh
./scripts/train/train_smolvlm_proassist.sh
```

### 3. Custom Training
```bash
python mmassist/train/finetune_smolvlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --use_lora true \
    --data_root_dir /path/to/proassist/data \
    --train_datasets wtag/dialog-klg_train_L0_I1 \
    --output_dir ./smolvlm-proassist-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

## Configuration Options

### Model Configuration
- `model_name_or_path`: SmolVLM model to fine-tune (default: "HuggingFaceTB/SmolVLM-Instruct")
- `use_lora`: Enable LoRA fine-tuning (recommended: true)
- `use_qlora`: Enable 4-bit quantization (for memory-limited setups)
- `freeze_vision`: Freeze vision encoder, train only language model

### Data Configuration
- `train_datasets`: ProAssist training datasets (e.g., "wtag/dialog-klg_train_L0_I1")
- `eval_datasets`: ProAssist evaluation datasets
- `frames_per_sample`: Maximum frames per training sample (reduce if memory issues)
- `use_4_1_aspect_ratio`: Resize images to 4:1 ratio for optimal encoding
- `max_seq_length`: Maximum sequence length (default: 2048)

### Training Configuration
- `per_device_train_batch_size`: Batch size per GPU (start with 1-2)
- `gradient_accumulation_steps`: Accumulate gradients for larger effective batch size
- `learning_rate`: Learning rate (default: 1e-4)
- `num_train_epochs`: Number of training epochs

## Memory Optimization

If you encounter GPU memory issues, try these optimizations in order:

1. **Reduce batch size**: Set `per_device_train_batch_size=1`
2. **Enable QLoRA**: Set `use_qlora=true` for 4-bit quantization
3. **Freeze vision**: Set `freeze_vision=true` to train only language components
4. **Reduce frames**: Set `frames_per_sample=1` or `2`
5. **Shorter sequences**: Reduce `max_seq_length` to 1024 or 1536

## Data Flow

```
ProAssist Dataset (temporal video conversations)
    ↓
ProAssistSmolVLMDataset (conversion layer)
    ↓ 
SmolVLM Chat Format (user/assistant with images)
    ↓
SmolVLM Processor (tokenization + image processing)
    ↓
Training Batch (input_ids + labels + pixel_values)
```

## Training Process

1. **Data Conversion**: Convert ProAssist conversations to SmolVLM chat format
2. **Frame Selection**: Sample frames from video segments (up to `frames_per_sample`)
3. **Image Processing**: Optionally resize to 4:1 aspect ratio for optimal encoding
4. **Chat Template**: Apply SmolVLM's chat template to create training text
5. **Tokenization**: Tokenize text and process images
6. **Label Creation**: Create labels for causal language modeling (mask image tokens)
7. **Training**: Fine-tune with LoRA/QLoRA

## Expected Output

The training will produce:
- Fine-tuned SmolVLM model adapted for ProAssist-style streaming assistance
- LoRA adapters (if using LoRA) for efficient deployment
- Training logs and metrics via tensorboard

## Monitoring Training

```bash
# View tensorboard logs
tensorboard --logdir ./smolvlm-proassist-finetune/runs

# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
tail -f nohup.out
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `per_device_train_batch_size` to 1
   - Enable `use_qlora=true`
   - Reduce `frames_per_sample`
   - Set `freeze_vision=true`

2. **Slow Training**
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Ensure `gradient_checkpointing=true` is not causing excessive slowdown
   - Consider using fewer frames per sample

3. **Poor Convergence**
   - Check learning rate (try 5e-5 or 2e-4)
   - Ensure sufficient training data
   - Verify data conversion is working correctly

4. **Data Loading Issues**
   - Run `test_smolvlm_training.py` to validate data pipeline
   - Check ProAssist dataset paths
   - Verify images are available in dataset

### Validation Commands

```bash
# Test data pipeline
python test_smolvlm_training.py

# Check dataset size
python -c "
from mmassist.data import build_train_dataset
ds = build_train_dataset(
    data_root_dir='/path/to/data',
    train_datasets='wtag/dialog-klg_train_L0_I1',
    keep_images=True
)
print(f'Dataset size: {len(ds)}')
"

# Test model loading
python -c "
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
model = Idefics3ForConditionalGeneration.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')
print('Model loaded successfully')
"
```

## Integration with ProAssist

After training, the fine-tuned SmolVLM can be used as a drop-in replacement in the ProAssist streaming inference pipeline. The model will be better adapted to:
- Understanding temporal video context
- Providing proactive assistance
- Generating appropriate responses for streaming scenarios

## Hardware Requirements

- **Minimum**: 1x GPU with 16GB VRAM (with QLoRA and optimizations)
- **Recommended**: 1x GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)
- **Memory**: 32GB+ system RAM
- **Storage**: 50GB+ free space for model, data, and checkpoints
