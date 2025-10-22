"""
Fine-tune SmolVLM-Instruct on ProAssist dataset for streaming video assistance.

This script adapts the SmolVLM fine-tuning approach to work with ProAssist's
multi-modal conversation format with temporal video frames.

LoRA Adapter Training & Combination:
------------------------------------
This script saves LoRA adapters that can be easily combined later. When using
--use_lora=True, only the adapter weights are saved, not the full model.

To train separate adapters on different datasets:
    python finetune_smolvlm.py --train_datasets dataset1 --output_dir ./adapter1
    python finetune_smolvlm.py --train_datasets dataset2 --output_dir ./adapter2

To combine multiple LoRA adapters after training:

    from peft import PeftModel
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    
    # Load base model
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct"
    )
    
    # Method 1: Load adapters sequentially (weighted average)
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    model = PeftModel.from_pretrained(base_model, "./adapter1")
    
    # Method 2: Use PEFT's weighted combination (requires PEFT >= 0.6.0)
    from peft import load_peft_weights
    adapters = ["./adapter1", "./adapter2"]
    weights = [0.5, 0.5]  # Equal weighting
    # Combine using PEFT utilities or manual weighted averaging
    
    # Method 3: Load multiple adapters and switch between them
    model = PeftModel.from_pretrained(base_model, "./adapter1", adapter_name="adapter1")
    model.load_adapter("./adapter2", adapter_name="adapter2")
    model.set_adapter("adapter1")  # Switch to adapter1
    
Each saved adapter directory contains:
    - adapter_config.json: LoRA configuration
    - adapter_model.safetensors (or .bin): Adapter weights
    - adapter_metadata.json: Training dataset info and hyperparameters
"""

import os
import torch
import logging
import pickle
import numpy as np
import psutil
import gc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Tuple
import transformers
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig

# ProAssist imports
from mmassist.configs import parse_args
from mmassist.data import build_train_dataset, build_eval_datasets
from mmassist.data.utils import tensor_to_pil_images
from mmassist.train.utils import is_global_rank_zero
from mmassist.train.proassist_smolvlm_dataset import ProAssistSmolVLMDataset


# W2T token constants
W2T_TOKEN_ID = 49191  # <|reserved_special_token_0|>
ASSISTANT_TOKEN_IDS = [9519, 9531, 42]  # "Assistant:"
END_OF_UTTERANCE_TOKEN_ID = 49279  # <end_of_utterance>
FAKE_TOKEN_AROUND_IMAGE_ID = 49189  # <fake_token_around_image>
IMAGE_TOKEN_ID = 49190  # <image>
GLOBAL_IMG_TOKEN_ID = 49152  # <global-img>

def log_memory_usage(stage="", logger=None):
    """Log current system memory usage."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # System memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        logger.info(f"{stage} - System Memory: {memory.percent:.1f}% used, "
                   f"{memory.available / 1024**3:.2f}GB available, "
                   f"{memory.used / 1024**3:.2f}GB used")
        
        if swap.total > 0:
            logger.info(f"{stage} - Swap Memory: {swap.percent:.1f}% used, "
                       f"{swap.used / 1024**3:.2f}GB used")

    except Exception as e:
        if logger:
            logger.warning(f"Failed to log memory usage: {e}")

def force_cleanup():
    """Force garbage collection and CUDA cache cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def log_gpu_utilization(stage="", logger=None):
    """Log GPU utilization using nvidia-ml-py if available."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            logger.info(f"{stage} - GPU {i}: {util.gpu}% util, {util.memory}% mem util, "
                       f"{memory_info.used/1024**3:.1f}GB/{memory_info.total/1024**3:.1f}GB, {temp}°C")
    except ImportError:
        # Fallback to basic torch info if pynvml not available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"{stage} - GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to log GPU utilization: {e}")

def get_smolvlm_learn_ranges(input_ids, frame_sampling_rate=1.0):
    """
    Get learning ranges for SmolVLM tokenized input.
    
    Args:
        input_ids: 1D tensor of token IDs
        frame_sampling_rate: Sampling rate for negative frames (0.0-1.0)
    
    Returns:
        List of (start_idx, end_idx, label_type) tuples where:
        - start_idx, end_idx: token position range
        - label_type: 'assistant' | 'w2t' | 'talk'
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    
    learn_ranges = []
    seq_len = len(input_ids)
    
    # Find assistant text ranges: [9519, 9531, 42] -> 49279
    i = 0
    while i < seq_len - 2:
        # Look for "Assistant:" token sequence
        if (input_ids[i] == ASSISTANT_TOKEN_IDS[0] and 
            input_ids[i+1] == ASSISTANT_TOKEN_IDS[1] and 
            input_ids[i+2] == ASSISTANT_TOKEN_IDS[2]):
            
            start_idx = i + 3  # Start after "Assistant:"
            
            # Find the end of assistant text (END_OF_UTTERANCE_TOKEN_ID)
            end_idx = start_idx
            while end_idx < seq_len and input_ids[end_idx] != END_OF_UTTERANCE_TOKEN_ID:
                end_idx += 1
            
            if end_idx < seq_len:
                # Include the end_of_utterance token in learning
                learn_ranges.append((start_idx, end_idx + 1, 'assistant'))
            i = end_idx
        else:
            i += 1
    
    # Find frame decision points by looking for pattern: <global-img><image>...<image><fake_token_around_image>
    # The last <fake_token_around_image> in each frame is where speaking decisions are made
    i = 0
    while i < seq_len:
        if input_ids[i] == GLOBAL_IMG_TOKEN_ID:
            # Skip any immediate <image> tokens to get to last <fake_token_around_image> of the frame
            while i < seq_len and input_ids[i] != FAKE_TOKEN_AROUND_IMAGE_ID:
                i += 1
                        
            # If next token is END_OF_UTTERANCE, this is a positive frame (talk)
            next_token = input_ids[i+1]
            if next_token == END_OF_UTTERANCE_TOKEN_ID:
                learn_ranges.append((i, i + 1, 'talk'))
            else:
                # This is a negative frame (don't talk) - sample based on sampling rate
                if frame_sampling_rate >= 1.0 or torch.rand(1).item() < frame_sampling_rate:
                    learn_ranges.append((i, i + 1, 'w2t'))
        
        i += 1
    
    return learn_ranges


@dataclass
class SmolVLMModelArguments:
    model_name_or_path: Optional[str] = field(
        default="HuggingFaceTB/SmolVLM-Instruct",
        metadata={"help": "Path to pretrained SmolVLM model"},
    )
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    use_qlora: bool = field(
        default=False, metadata={"help": "Whether to use QLoRA (4-bit quantization)"}
    )
    freeze_vision: bool = field(
        default=False,
        metadata={"help": "Whether to freeze vision encoder during training"},
    )


@dataclass
class SmolVLMDataArguments:
    train_datasets: str = field(
        default="wtag/dialog-klg-sum_train_L2048_I1",
        metadata={"help": "Training datasets to use"},
    )
    eval_datasets: Optional[str] = field(
        default="wtag/dialog-klg-sum_val_L2048_I1",
        metadata={"help": "Evaluation datasets to use"},
    )
    data_root_dir: str = field(
        default="/projects/beto/swong2/proassist_data/processed_data",
        metadata={"help": "Root directory for ProAssist data"},
    )
    use_4_1_aspect_ratio: bool = field(
        default=True,
        metadata={
            "help": "Whether to use 4:1 aspect ratio for optimal SmolVLM encoding"
        },
    )
    frame_sampling_ratio: float = field(
        default=0.1, metadata={"help": "Ratio of frames to sample from frame ranges"}
    )
    w2t_frame_sampling_rate: float = field(
        default=0.3, metadata={"help": "Sampling rate for negative frames in w2t learning (0.0-1.0)"}
    )
    use_end_of_utterance_for_w2t: bool = field(
        default=False, 
        metadata={"help": "If True, use END_OF_UTTERANCE_TOKEN_ID (49279) as W2T token instead of reserved_special_token_0 (49191)"}
    )
    w2t_only: bool = field(
        default=False,
        metadata={"help": "If True, only compute loss for w2t tokens (ignore assistant text). Useful for focusing training on speaking decisions."}
    )
    no_assistant: bool = field(
        default=False,
        metadata={"help": "If True, disable learning from assistant tokens (but keep talk tokens). Useful for focusing on speaking decisions while still learning from positive frames."}
    )
    context_size_limit: int = field(
        default=7500,
        metadata={
            "help": "Context size limit in tokens before splitting samples (leave room below 8k)"
        },
    )


@dataclass
class SmolVLMTrainingArguments(TrainingArguments):
    output_dir: str = field(default="/work/nvme/beto/swong2/smolvlm_proassist_finetune")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=250)
    save_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    save_total_limit: int = field(default=10)
    optim: str = field(default="paged_adamw_8bit") # or adamw_torch_fused ideally
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False) # saves memory; recompute gradients during backward pass; disable for lora
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of workers for data loading. 0 means main process only."})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin memory in data loaders for faster GPU transfer."})
    dataloader_prefetch_factor: Optional[int] = field(default=None, metadata={"help": "Number of batches loaded in advance by each worker. None means 2 if num_workers > 0."})
    w2t_loss_weight: float = field(default=0.5, metadata={"help": "Weight for w2t loss (0.0-1.0). Assistant loss weight is inferred as 1 - w2t_loss_weight. Mutually exclusive with use_inverse_freq_weighting."})
    use_inverse_freq_weighting: bool = field(default=False, metadata={"help": "If True, use inverse frequency weighting per batch for label types. Mutually exclusive with w2t_loss_weight (which should be set to 0.5 when this is enabled)."})


@dataclass
class LoraArguments:
    lora_r: int = field(default=16) # increase if model underfits
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1) # increase if model overfits
    lora_target_modules: str = field(
        default="down_proj,o_proj,k_proj,q_proj,gate_proj,up_proj,v_proj"
    )
    use_dora: bool = field(default=False) # better performance usually but slower


def collate_fn(examples, processor, w2t_frame_sampling_rate=0.3, w2t_token_id=None, w2t_only=False, no_assistant=False):
    """Enhanced collate function for SmolVLM training with w2t token support.
    
    Args:
        examples: List of dataset examples
        processor: The model processor
        w2t_frame_sampling_rate: Sampling rate for negative frames (0.0-1.0)
        w2t_token_id: Token ID to use for w2t predictions
        w2t_only: If True, only compute loss for w2t tokens (ignore assistant text and talk tokens)
        no_assistant: If True, disable learning from assistant tokens (but keep talk tokens)
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Default to reserved_special_token_0 if not specified
    if w2t_token_id is None:
        w2t_token_id = W2T_TOKEN_ID
    
    # Log memory at start of collate_fn
    log_memory_usage(f"Collate start with {len(examples)} examples", logger)
    
    # Filter out None examples
    examples = [ex for ex in examples if ex is not None]

    if not examples:
        return None

    texts = []
    all_images = []
    
    # Track samples with and without images for logging
    samples_with_images = 0
    samples_without_images = 0
    
    # Time the data preparation phase
    prep_start = time.time()

    for example in examples:
        messages = example["messages"]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False
        )
        texts.append(text.strip())

        # Collect all images from this conversation
        images = example.get("images", [])
        
        # Track image statistics
        if images and len(images) > 0:
            samples_with_images += 1
        else:
            samples_without_images += 1
            logger.debug(f"Sample has no images (text-only sample)")
        
        # Debug: Log image information
        logger.debug(f"Sample has {len(images) if images else 0} images, text length: {len(text)}")

        # # Count and validate image tokens
        # image_token_count = text.count('<image>')        
        # print(f"Sample has {image_token_count} <image> tokens and {len(images)} actual images")

        # # Count tokens
        # inputs = processor(
        #     text=text,
        #     images=images,
        #     return_tensors="pt",
        # )
        # token_count = inputs["input_ids"].shape[1]
        # print(f"Sample's total token count: {token_count}")
        # num_image_tokens = (inputs["input_ids"] == IMAGE_TOKEN_ID).sum().item()
        # print(f"Sample's image token count: {num_image_tokens}")
        

        all_images.append(images)
        
        # Force cleanup every few samples to prevent memory buildup
        if len(all_images) % 5 == 0:
            force_cleanup()

    prep_time = time.time() - prep_start
    logger.info(f"Collate data preparation took {prep_time:.3f}s for {len(examples)} examples")
    
    # Log image statistics
    if samples_without_images > 0:
        logger.info(f"Batch has {samples_with_images} samples with images and {samples_without_images} text-only samples")
    
    # Debug: Print batch content before processor
    logger.debug(f"Batch size: {len(texts)} texts, {len(all_images)} image lists")
    for i, images in enumerate(all_images):
        if not images or len(images) == 0:
            logger.debug(f"  Item {i}: text-only (no images)")
        else:
            logger.debug(f"  Item {i}: {len(images)} images")
    
    # Time the processor call
    processor_start = time.time()
    
    # Separate processing for batches with and without images
    # If all samples have no images, don't pass images parameter
    has_any_images = any(img_list for img_list in all_images if img_list)
    
    if has_any_images:
        # Process batch with images (some samples may still be text-only)
        batch = processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
    else:
        # All samples are text-only, process without images parameter
        logger.info("Processing text-only batch (no images in any sample)")
        batch = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
    
    processor_time = time.time() - processor_start
    logger.info(f"Processor call took {processor_time:.3f}s, output shape: {batch['input_ids'].shape}")
    
    # Time the label creation
    label_start = time.time()

    # Create labels with advanced w2t masking
    labels = torch.full_like(batch["input_ids"], -100, dtype=torch.long)
    
    # Create label_types tensor: 0=ignore, 1=assistant, 2=w2t, 3=talk
    label_types = torch.zeros_like(batch["input_ids"], dtype=torch.long)

    # Process each sample in the batch
    for i, input_ids in enumerate(batch["input_ids"]):
        # Get learning ranges for this sample
        learn_ranges = get_smolvlm_learn_ranges(
            input_ids, frame_sampling_rate=w2t_frame_sampling_rate
        )
        
        # Apply masking based on range types
        for start_idx, end_idx, label_type in learn_ranges:
            if label_type == 'assistant':
                # Learn from actual assistant tokens (skip if w2t_only mode or no_assistant mode)
                if not no_assistant:
                    labels[i, start_idx:end_idx] = input_ids[start_idx:end_idx]
                    label_types[i, start_idx:end_idx] = 1  # Mark as assistant tokens
            elif label_type == 'w2t':
                # Learn to predict w2t token at frame decision point
                labels[i, start_idx] = w2t_token_id
                label_types[i, start_idx] = 2  # Mark as w2t token
            elif label_type == 'talk':
                # Learn to predict the token after <end_of_utterance> (positive frame)
                # At talk positions: <fake_token_around_image> -> <end_of_utterance> -> next_token
                # We want to predict next_token, not <end_of_utterance> (which is what w2t predicts)
                if end_idx + 1 < len(input_ids):
                    labels[i, start_idx] = input_ids[end_idx + 1]
                    label_types[i, start_idx] = 3  # Mark as talk token

    # Mask padding tokens
    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens (model shouldn't predict image tokens)
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    labels[labels == image_token_id] = -100

    # VALIDATION: Check that w2t and talk positions have correct target tokens
    # W2T positions should predict END_OF_UTTERANCE (or w2t_token_id)
    # Talk positions should NOT predict END_OF_UTTERANCE
    w2t_positions_mask = (label_types == 2)
    talk_positions_mask = (label_types == 3)
    
    if w2t_positions_mask.any():
        w2t_labels = labels[w2t_positions_mask]
        # Check if w2t positions are predicting the correct token
        expected_w2t_token = w2t_token_id
        incorrect_w2t = (w2t_labels != expected_w2t_token) & (w2t_labels != -100)
        if incorrect_w2t.any():
            num_incorrect = incorrect_w2t.sum().item()
            total_w2t = (w2t_labels != -100).sum().item()
            logger.warning(
                f"W2T VALIDATION WARNING: {num_incorrect}/{total_w2t} w2t positions are NOT predicting "
                f"the expected w2t token (ID: {expected_w2t_token}). "
                f"This should not happen!"
            )
    
    if talk_positions_mask.any():
        talk_labels = labels[talk_positions_mask]
        # Check if talk positions are NOT predicting END_OF_UTTERANCE
        predicting_eou = (talk_labels == END_OF_UTTERANCE_TOKEN_ID) & (talk_labels != -100)
        if predicting_eou.any():
            num_predicting_eou = predicting_eou.sum().item()
            total_talk = (talk_labels != -100).sum().item()
            logger.warning(
                f"TALK VALIDATION WARNING: {num_predicting_eou}/{total_talk} talk positions are predicting "
                f"<end_of_utterance> (ID: {END_OF_UTTERANCE_TOKEN_ID}). "
                f"Talk positions should predict the token AFTER <end_of_utterance>, not <end_of_utterance> itself!"
            )

    batch["labels"] = labels
    batch["label_types"] = label_types
    
    label_time = time.time() - label_start
    total_time = time.time() - start_time
    logger.info(f"Label creation took {label_time:.3f}s, total collate_fn time: {total_time:.3f}s")
    
    # Log memory at end of collate_fn
    log_memory_usage("Collate end", logger)
    
    # Clean up before returning
    force_cleanup()
    
    # Mark the timestamp when collate_fn finishes
    batch["_collate_end_time"] = time.time()

    return batch


class TimedDataLoader:
    """Wrapper around DataLoader to log batch retrieval timing."""
    def __init__(self, dataloader, logger):
        self.dataloader = dataloader
        self.logger = logger
        self.batch_count = 0
    
    def __iter__(self):
        self.batch_count = 0
        for batch in self.dataloader:
            self.batch_count += 1
            fetch_time = time.time()
            self.logger.info(f">>> BATCH {self.batch_count} RETRIEVED from dataloader at {fetch_time:.3f}")
            yield batch
    
    def __len__(self):
        return len(self.dataloader)


class SmolVLMProAssistTrainer(Trainer):
    """Custom trainer for SmolVLM ProAssist fine-tuning."""

    def __init__(self, processor, w2t_frame_sampling_rate=0.3, w2t_token_id=None, w2t_only=False, no_assistant=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.w2t_frame_sampling_rate = w2t_frame_sampling_rate
        self.w2t_token_id = w2t_token_id if w2t_token_id is not None else W2T_TOKEN_ID
        self.w2t_only = w2t_only
        self.no_assistant = no_assistant
        self.step_count = 0
        self.logger = logging.getLogger(__name__)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss based on label types (assistant vs w2t).
        
        Two weighting strategies are available (mutually exclusive):
        1. Fixed weights: w2t_loss_weight and (1 - w2t_loss_weight) for assistant
        2. Inverse frequency weighting: dynamically computed per batch based on token counts
        
        This implementation replicates HuggingFace Trainer's default causal LM loss
        computation but adds custom weighting based on label types.
        """
        # Validate that only one weighting strategy is used
        if self.args.use_inverse_freq_weighting and self.args.w2t_loss_weight != 0.5:
            raise ValueError(
                "use_inverse_freq_weighting and w2t_loss_weight are mutually exclusive. "
                "When using inverse frequency weighting, w2t_loss_weight should be set to 0.5 (default)."
            )
        
        # Extract label_types if present
        label_types = inputs.pop("label_types", None)
        
        # VERIFICATION: Compute baseline loss using default HF method for comparison
        # Only verify during training (not during eval) to avoid repeated logs during eval batches
        verify_loss_computation = (self.step_count % 100 == 0) and model.training
        baseline_loss = None
        
        if verify_loss_computation and label_types is not None:
            # Compute what the loss would be with default HF implementation
            with torch.no_grad():
                baseline_outputs = model(**inputs)
                if hasattr(baseline_outputs, "loss") and baseline_outputs.loss is not None:
                    baseline_loss = baseline_outputs.loss.item()
        
        # Get the model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        if labels is None:
            raise ValueError("Labels must be provided for loss computation")
        
        # Compute standard cross-entropy loss without reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Shift logits and labels for causal LM (standard for autoregressive models)
        # This matches HuggingFace's implementation in modeling_utils.py
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)
        
        # Compute per-token loss
        per_token_loss = loss_fct(shift_logits_flat, shift_labels_flat)
        per_token_loss = per_token_loss.view(shift_labels.size())
        
        # Create mask for valid (non-ignored) tokens
        mask = (shift_labels != -100).float()
        
        # Check if we have any valid tokens
        num_valid_tokens = mask.sum()
        if num_valid_tokens == 0:
            self.logger.warning(f"No valid tokens in batch at step {self.step_count}")
            # Return zero loss if no valid tokens (shouldn't happen in practice)
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss
        
        # VERIFICATION: Compute unweighted loss (should match HF default)
        unweighted_loss = (per_token_loss * mask).sum() / num_valid_tokens
        
        # Apply weighted loss if label_types provided
        if label_types is not None:
            # Shift label_types to match shifted labels
            shift_label_types = label_types[..., 1:].contiguous()
            
            # Initialize weights to 1.0 (for ignored tokens, will be masked out anyway)
            weights = torch.ones_like(shift_label_types, dtype=per_token_loss.dtype)
            
            if self.args.use_inverse_freq_weighting:
                # Compute inverse frequency weights dynamically per batch
                num_assistant = ((shift_label_types == 1) & (shift_labels != -100)).sum().float()
                num_w2t = ((shift_label_types == 2) & (shift_labels != -100)).sum().float()
                num_talk = ((shift_label_types == 3) & (shift_labels != -100)).sum().float()
                
                total_labeled = num_assistant + num_w2t + num_talk
                
                if total_labeled > 0:
                    # Compute inverse frequency weights
                    # Add small epsilon to avoid division by zero
                    eps = 1e-8
                    inv_freq_assistant = total_labeled / (num_assistant + eps)
                    inv_freq_w2t = total_labeled / (num_w2t + eps)
                    inv_freq_talk = total_labeled / (num_talk + eps)
                    
                    # Normalize weights so they average to 1.0
                    weight_sum = inv_freq_assistant + inv_freq_w2t + inv_freq_talk
                    inv_freq_assistant = inv_freq_assistant / weight_sum * 3.0
                    inv_freq_w2t = inv_freq_w2t / weight_sum * 3.0
                    inv_freq_talk = inv_freq_talk / weight_sum * 3.0
                    
                    # Apply inverse frequency weights
                    weights[shift_label_types == 1] = inv_freq_assistant
                    weights[shift_label_types == 2] = inv_freq_w2t
                    weights[shift_label_types == 3] = inv_freq_talk
                    
                    w2t_weight = inv_freq_w2t.item()
                    assistant_weight = inv_freq_assistant.item()
                    talk_weight = inv_freq_talk.item()
                else:
                    w2t_weight = 1.0
                    assistant_weight = 1.0
                    talk_weight = 1.0
            else:
                # Use fixed weights from w2t_loss_weight parameter
                w2t_weight = self.args.w2t_loss_weight
                assistant_weight = 1.0 - w2t_weight
                talk_weight = w2t_weight
                
                # Apply weights based on label type
                # label_types: 0=ignore, 1=assistant, 2=w2t, 3=talk
                weights[shift_label_types == 1] = assistant_weight  # assistant tokens
                weights[shift_label_types == 2] = w2t_weight  # w2t tokens
                weights[shift_label_types == 3] = talk_weight  # talk tokens (same as w2t tokens)
            
            # VERIFICATION: Count tokens by type for logging
            if verify_loss_computation:
                num_assistant_tokens = (shift_label_types == 1).sum().item()
                num_w2t_tokens = (shift_label_types == 2).sum().item()
                num_talk_tokens = (shift_label_types == 3).sum().item()
                num_valid_tokens = mask.sum().item()
                
                # Compute separate losses for each type
                assistant_mask = ((shift_label_types == 1) & (shift_labels != -100)).float()
                w2t_mask = ((shift_label_types == 2) & (shift_labels != -100)).float()
                talk_mask = ((shift_label_types == 3) & (shift_labels != -100)).float()
                
                assistant_loss = (per_token_loss * assistant_mask).sum() / assistant_mask.sum() if assistant_mask.sum() > 0 else 0.0
                w2t_loss = (per_token_loss * w2t_mask).sum() / w2t_mask.sum() if w2t_mask.sum() > 0 else 0.0
                talk_loss = (per_token_loss * talk_mask).sum() / talk_mask.sum() if talk_mask.sum() > 0 else 0.0
                
                self.logger.info("=" * 60)
                self.logger.info(f"LOSS COMPUTATION VERIFICATION (Step {self.step_count})")
                self.logger.info("=" * 60)
                self.logger.info(f"Token counts:")
                self.logger.info(f"  Total valid tokens: {num_valid_tokens}")
                self.logger.info(f"  Assistant tokens: {num_assistant_tokens} ({num_assistant_tokens/num_valid_tokens*100:.1f}%)")
                self.logger.info(f"  W2T tokens: {num_w2t_tokens} ({num_w2t_tokens/num_valid_tokens*100:.1f}%)")
                self.logger.info(f"  Talk tokens: {num_talk_tokens} ({num_talk_tokens/num_valid_tokens*100:.1f}%)")
                self.logger.info(f"Loss breakdown:")
                self.logger.info(f"  Unweighted total loss: {unweighted_loss.item():.6f}")
                self.logger.info(f"  Assistant average loss (unweighted): {assistant_loss:.6f}")
                self.logger.info(f"  W2T average loss (unweighted): {w2t_loss:.6f}")
                self.logger.info(f"  Talk average loss (unweighted): {talk_loss:.6f}")
                self.logger.info(f"Weighting scheme:")
                if self.args.use_inverse_freq_weighting:
                    self.logger.info(f"  Using inverse frequency weighting (per-batch dynamic)")
                else:
                    self.logger.info(f"  Using fixed weighting (w2t_loss_weight={self.args.w2t_loss_weight})")
                self.logger.info(f"  Assistant weight: {assistant_weight:.2f}")
                self.logger.info(f"  W2T weight: {w2t_weight:.2f}")
                self.logger.info(f"  Talk weight: {talk_weight:.2f}")
                if baseline_loss is not None:
                    self.logger.info(f"HuggingFace baseline loss: {baseline_loss:.6f}")
                    loss_diff = abs(unweighted_loss.item() - baseline_loss)
                    if loss_diff > 1e-4:
                        self.logger.warning(f"  WARNING: Unweighted loss differs from HF baseline by {loss_diff:.6f}")
                    else:
                        self.logger.info(f"  ✓ Unweighted loss matches HF baseline (diff: {loss_diff:.2e})")
            
            # Apply weights to loss
            weighted_loss = per_token_loss * weights
            
            # Mask out ignored tokens (where shift_labels == -100)
            weighted_loss = weighted_loss * mask
            
            # Compute mean loss over non-ignored tokens
            num_valid_tokens_weighted = mask.sum()
            if num_valid_tokens_weighted == 0:
                self.logger.warning(f"No valid tokens after weighting at step {self.step_count}")
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss = weighted_loss.sum() / num_valid_tokens_weighted
            
            # VERIFICATION: Log weighted loss
            if verify_loss_computation:
                self.logger.info(f"  Final weighted loss: {loss.item():.6f}")
                weighted_change = ((loss.item() - unweighted_loss.item()) / unweighted_loss.item()) * 100
                self.logger.info(f"  Change from unweighted: {weighted_change:+.2f}%")
                self.logger.info("=" * 60)
        else:
            # Fallback to standard mean loss if no label_types
            loss = unweighted_loss
            shift_label_types = None  # Set to None for error logging
            
            if verify_loss_computation:
                self.logger.info(f"Step {self.step_count}: Using unweighted loss (no label_types): {loss.item():.6f}")
        
        # SANITY CHECKS
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"Invalid loss detected at step {self.step_count}: {loss.item()}")
            self.logger.error(f"  Model training mode: {model.training}")
            self.logger.error(f"  Logits shape: {logits.shape}")
            self.logger.error(f"  Labels shape: {labels.shape}")
            self.logger.error(f"  Valid tokens: {num_valid_tokens.item() if isinstance(num_valid_tokens, torch.Tensor) else num_valid_tokens}")
            self.logger.error(f"  Unweighted loss: {unweighted_loss.item()}")
            
            # Check logits and labels for issues
            self.logger.error(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            self.logger.error(f"  Logits NaN count: {torch.isnan(logits).sum().item()}")
            self.logger.error(f"  Logits Inf count: {torch.isinf(logits).sum().item()}")
            
            # Check per_token_loss for issues
            self.logger.error(f"  Per-token loss stats: min={per_token_loss.min().item():.4f}, max={per_token_loss.max().item():.4f}")
            self.logger.error(f"  Per-token loss NaN count: {torch.isnan(per_token_loss).sum().item()}")
            self.logger.error(f"  Per-token loss Inf count: {torch.isinf(per_token_loss).sum().item()}")
            
            if label_types is not None and shift_label_types is not None:
                self.logger.error(f"  Label types provided: True")
                self.logger.error(f"  Assistant tokens: {(shift_label_types == 1).sum().item()}")
                self.logger.error(f"  W2T tokens: {(shift_label_types == 2).sum().item()}")
                self.logger.error(f"  Talk tokens: {(shift_label_types == 3).sum().item()}")
            else:
                self.logger.error(f"  Label types provided: False")
            
            raise ValueError(f"Loss is {loss.item()}, training cannot continue")
        
        return (loss, outputs) if return_outputs else loss


    def get_train_dataloader(self):
        """Override to use custom collate function with timing wrapper."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        from functools import partial
        from torch.utils.data import DataLoader

        collate_fn_with_processor = partial(
            collate_fn, 
            processor=self.processor,
            w2t_frame_sampling_rate=self.w2t_frame_sampling_rate,
            w2t_token_id=self.w2t_token_id,
            w2t_only=self.w2t_only,
            no_assistant=self.no_assistant
        )

        dataloader_kwargs = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": True,
            "collate_fn": collate_fn_with_processor,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        
        # Add prefetch_factor only if num_workers > 0
        if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        base_dataloader = DataLoader(self.train_dataset, **dataloader_kwargs)
        
        # Wrap dataloader to log when batches are retrieved
        return TimedDataLoader(base_dataloader, self.logger)

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use custom collate function."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        from functools import partial
        from torch.utils.data import DataLoader

        collate_fn_with_processor = partial(
            collate_fn, 
            processor=self.processor,
            w2t_frame_sampling_rate=self.w2t_frame_sampling_rate,
            w2t_token_id=self.w2t_token_id,
            w2t_only=self.w2t_only,
            no_assistant=self.no_assistant
        )

        dataloader_kwargs = {
            "batch_size": self.args.per_device_eval_batch_size,
            "shuffle": False,
            "collate_fn": collate_fn_with_processor,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        
        # Add prefetch_factor only if num_workers > 0
        if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        return DataLoader(eval_dataset, **dataloader_kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to add detailed timing and monitoring."""
        step_start_time = time.time()
        self.step_count += 1
        
        # Check if we have collate timing info
        collate_end_time = inputs.pop("_collate_end_time", None)
        if collate_end_time is not None:
            gap = step_start_time - collate_end_time
            self.logger.info(f">>> GAP between collate_fn end and training_step start: {gap:.3f}s")
        
        # Log detailed info every 5 steps
        if self.step_count % 5 == 0:
            self.logger.info(f"=== Training Step {self.step_count} START ===")
            log_memory_usage(f"Step {self.step_count} start", self.logger)
            log_gpu_utilization(f"Step {self.step_count} start", self.logger)
        
        # Time the forward and backward pass (includes GPU transfer)
        forward_start = time.time()
        
        # Call parent training step (this handles GPU transfer internally)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        forward_time = time.time() - forward_start
        total_step_time = time.time() - step_start_time
        
        # Log timing every 10 steps
        if self.step_count % 10 == 0:
            self.logger.info(f"Step {self.step_count}: forward+backward alone took {forward_time:.3f}s, total step: {total_step_time:.3f}s")
            log_gpu_utilization(f"Step {self.step_count} end", self.logger)
            self.logger.info(f"=== Training Step {self.step_count} END ===")
                
        # Force cleanup every 50 steps
        if self.step_count % 50 == 0:
            cleanup_start = time.time()
            force_cleanup()
            cleanup_time = time.time() - cleanup_start
            self.logger.info(f"Cleanup at step {self.step_count} took {cleanup_time:.3f}s")
            log_memory_usage(f"After cleanup step {self.step_count}", self.logger)
        
        return loss

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation. This is required for Trainer to run evaluation.
        We'll compute basic metrics here, and speaking decision metrics separately.
        """
        # Return empty dict - we'll compute detailed metrics in evaluate() override
        return {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to include both standard metrics and speaking decision evaluation.
        """
        self.logger.info("=" * 50)
        self.logger.info(f"EVALUATION STARTING at step {self.state.global_step}")
        self.logger.info("=" * 50)
        
        # Clean memory before evaluation
        force_cleanup()
        log_memory_usage("Before standard eval", self.logger)
        
        # First run standard evaluation (computes eval_loss)
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        self.logger.info(f"Standard evaluation complete. Loss: {metrics.get(f'{metric_key_prefix}_loss', 'N/A')}")
        
        # Clean memory between evaluation passes
        force_cleanup()
        log_memory_usage("After standard eval, before speaking eval", self.logger)
        
        # Only run speaking decision evaluation every N evals to save memory
        # Or skip it entirely by setting skip_speaking_eval=True
        skip_speaking_eval = False  # Set to True to skip and save memory
        eval_speaking_every_n = 1  # Only evaluate speaking decisions every N evals (1 = every time)
        
        speaking_metrics = {}
        if not skip_speaking_eval and (self.state.global_step // self.args.eval_steps) % eval_speaking_every_n == 0:
            # Then run speaking decision evaluation
            self.logger.info("Starting speaking decision evaluation...")
            speaking_metrics = self.evaluate_speaking_decisions(eval_dataset)
        else:
            self.logger.info("Skipping speaking decision evaluation this step to save memory")
        
        # Merge the metrics
        if speaking_metrics:
            # Add prefix to speaking metrics for tensorboard
            prefixed_speaking_metrics = {
                f"{metric_key_prefix}_{k}": v for k, v in speaking_metrics.items()
            }
            metrics.update(prefixed_speaking_metrics)
            self.logger.info(f"Speaking decision metrics added: {list(prefixed_speaking_metrics.keys())}")
            
            # Explicitly log to tensorboard
            if self.args.report_to and "tensorboard" in self.args.report_to:
                for key, value in prefixed_speaking_metrics.items():
                    self.log({key: value})
        
        self.logger.info("=" * 50)
        self.logger.info(f"EVALUATION COMPLETE at step {self.state.global_step}")
        self.logger.info("=" * 50)
        
        return metrics

    def evaluate_speaking_decisions(self, eval_dataset=None):
        """
        Evaluate w2t probability statistics at speaking decision positions.
        
        Returns:
            dict: Evaluation metrics including w2t probability mean/variance for:
                  - w2t positions (negative frames)
                  - talk positions (positive frames)
                  - both positions combined
        """
        import torch.nn.functional as F
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        logger = logging.getLogger(__name__)
        logger.info("Starting speaking decision evaluation...")
        
        self.model.eval()
        
        w2t_position_probs = []  # W2T probabilities at w2t positions
        talk_position_probs = []  # W2T probabilities at talk positions
        total_w2t_positions = 0
        total_talk_positions = 0
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                if batch is None:
                    continue
                    
                # Move batch to device
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                pixel_values = batch["pixel_values"].to(self.model.device) if "pixel_values" in batch else None
                
                # Get model predictions
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # Process each sample in the batch
                for i, sample_input_ids in enumerate(input_ids):
                    # Get learning ranges for this sample (same as training)
                    learn_ranges = get_smolvlm_learn_ranges(
                        sample_input_ids, 
                        frame_sampling_rate=self.w2t_frame_sampling_rate
                    )
                    
                    # Extract speaking decision positions and w2t probabilities
                    for start_idx, end_idx, label_type in learn_ranges:
                        if label_type in ['w2t', 'talk']:
                            if start_idx < logits.shape[1]:
                                # Get prediction at decision point
                                position_logits = logits[i, start_idx]  # [vocab_size]
                                
                                # Get probabilities for W2T vs other tokens
                                probs = F.softmax(position_logits, dim=-1)
                                w2t_prob = probs[self.w2t_token_id].item()
                                
                                # Track w2t probability by position type
                                if label_type == 'w2t':
                                    w2t_position_probs.append(w2t_prob)
                                    total_w2t_positions += 1
                                else:  # 'talk'
                                    talk_position_probs.append(w2t_prob)
                                    total_talk_positions += 1
                
                # Clean up batch tensors to free memory
                del input_ids, attention_mask, pixel_values, outputs, logits
                
                # Force cleanup every 5 batches during eval
                if batch_idx % 5 == 0:
                    force_cleanup()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(eval_dataloader)}")
                    log_memory_usage(f"Eval batch {batch_idx}", logger)
        
        if not w2t_position_probs and not talk_position_probs:
            logger.warning("No speaking decision positions found in evaluation data")
            return {}
        
        # Calculate w2t probability statistics for different position types
        metrics = {
            'total_w2t_positions': int(total_w2t_positions),
            'total_talk_positions': int(total_talk_positions),
            'total_decision_points': int(total_w2t_positions + total_talk_positions),
        }
        
        # Statistics for w2t positions (negative frames - should have high w2t prob)
        if w2t_position_probs:
            w2t_pos_array = np.array(w2t_position_probs)
            metrics['w2t_pos_prob_mean'] = float(np.mean(w2t_pos_array))
            metrics['w2t_pos_prob_variance'] = float(np.var(w2t_pos_array))
        
        # Statistics for talk positions (positive frames - should have low w2t prob)
        if talk_position_probs:
            talk_pos_array = np.array(talk_position_probs)
            metrics['talk_pos_prob_mean'] = float(np.mean(talk_pos_array))
            metrics['talk_pos_prob_variance'] = float(np.var(talk_pos_array))
        
        # Statistics for both positions combined
        all_probs = w2t_position_probs + talk_position_probs
        if all_probs:
            all_probs_array = np.array(all_probs)
            metrics['both_pos_prob_mean'] = float(np.mean(all_probs_array))
            metrics['both_pos_prob_variance'] = float(np.var(all_probs_array))
        
        logger.info("Speaking Decision Evaluation Results:")
        logger.info(f"  Total decision points: {metrics['total_decision_points']}")
        logger.info(f"  W2T positions: {total_w2t_positions}")
        logger.info(f"  Talk positions: {total_talk_positions}")
        
        if w2t_position_probs:
            logger.info(f"  W2T Positions - W2T Prob Mean: {metrics['w2t_pos_prob_mean']:.4f}")
            logger.info(f"  W2T Positions - W2T Prob Variance: {metrics['w2t_pos_prob_variance']:.4f}")
        
        if talk_position_probs:
            logger.info(f"  Talk Positions - W2T Prob Mean: {metrics['talk_pos_prob_mean']:.4f}")
            logger.info(f"  Talk Positions - W2T Prob Variance: {metrics['talk_pos_prob_variance']:.4f}")
        
        if all_probs:
            logger.info(f"  Both Positions - W2T Prob Mean: {metrics['both_pos_prob_mean']:.4f}")
            logger.info(f"  Both Positions - W2T Prob Variance: {metrics['both_pos_prob_variance']:.4f}")
        
        self.model.train()  # Reset to training mode
        return metrics


def setup_model_and_processor(
    model_args: SmolVLMModelArguments, lora_args: LoraArguments
):
    """Setup SmolVLM model and processor with optional LoRA."""
    
    logger = logging.getLogger(__name__)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # Setup quantization if using QLoRA
    quantization_config = None
    if model_args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze vision encoder if requested
    if model_args.freeze_vision:
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
        if is_global_rank_zero():
            logger.info("Vision encoder frozen")

    # Setup LoRA if requested
    if model_args.use_lora or model_args.use_qlora:
        target_modules = lora_args.lora_target_modules.split(",")

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            use_dora=lora_args.use_dora and not model_args.use_qlora,
            task_type="CAUSAL_LM",
        )

        if model_args.use_qlora:
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, lora_config)
        
        # Enable input require grads for compatibility with gradient checkpointing
        # This is necessary when using LoRA with gradient checkpointing
        model.enable_input_require_grads()

        if is_global_rank_zero():
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)"
            )

    return model, processor


def main():

    parser = HfArgumentParser(
        (
            SmolVLMModelArguments,
            SmolVLMDataArguments,
            SmolVLMTrainingArguments,
            LoraArguments,
        )
    )

    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # Set up logging
    log_file = os.path.join(training_args.output_dir, "training.log")
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Keep console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}") # make sure this lists all gpus
    
    # Log initial memory state
    log_memory_usage("Initial state", logger)

    if is_global_rank_zero():
        logger.info("=" * 50)
        logger.info("SmolVLM ProAssist Fine-tuning")
        logger.info("=" * 50)
        logger.info(f"Model: {model_args.model_name_or_path}")
        logger.info(f"Use LoRA: {model_args.use_lora}")
        logger.info(f"Use QLoRA: {model_args.use_qlora}")
        logger.info(f"Freeze Vision: {model_args.freeze_vision}")
        logger.info(f"Train datasets: {data_args.train_datasets}")
        logger.info(f"Eval datasets: {data_args.eval_datasets}")
        logger.info(f"4:1 aspect ratio: {data_args.use_4_1_aspect_ratio}")
        logger.info(f"Frame sampling ratio: {data_args.frame_sampling_ratio}")
        logger.info(f"W2T frame sampling rate: {data_args.w2t_frame_sampling_rate}")
        if training_args.use_inverse_freq_weighting:
            logger.info(f"Using inverse frequency weighting (per-batch dynamic)")
        else:
            logger.info(f"W2T loss weight: {training_args.w2t_loss_weight}")
            logger.info(f"Assistant loss weight: {1.0 - training_args.w2t_loss_weight}")
        logger.info(f"Context size limit: {data_args.context_size_limit}")

    # Setup model and processor
    logger.info("Setting up model and processor...")
    model_setup_start = time.time()
    model, processor = setup_model_and_processor(model_args, lora_args)
    model_setup_time = time.time() - model_setup_start
    logger.info(f"Model and processor setup took {model_setup_time:.2f}s")
    
    # Log memory after model setup
    log_memory_usage("After model setup", logger)
    log_gpu_utilization("After model setup", logger)

    # Build ProAssist datasets
    all_args_dict = {
        "data_root_dir": data_args.data_root_dir,
        "train_datasets": data_args.train_datasets,
        "eval_datasets": data_args.eval_datasets,
        "print_info": is_global_rank_zero(),
        "keep_images": True,  # Essential for SmolVLM training
        "remove_summarize_turns": False,
    }

    if is_global_rank_zero():
        logger.info("Loading ProAssist datasets...")

    dataset_start = time.time()
    train_dataset = build_train_dataset(**all_args_dict)
    train_dataset_time = time.time() - dataset_start
    logger.info(f"Loading train dataset took {train_dataset_time:.2f}s")
    
    eval_start = time.time()
    eval_datasets = (
        build_eval_datasets(**all_args_dict) if data_args.eval_datasets else {}
    )
    eval_dataset_time = time.time() - eval_start
    logger.info(f"Loading eval dataset took {eval_dataset_time:.2f}s")

    from torch.utils.data import Subset

    # Convert to SmolVLM format
    logger.info("Initializing SmolVLM train dataset...")
    smolvlm_train_start = time.time()
    smolvlm_train_dataset = ProAssistSmolVLMDataset(
        # Subset(train_dataset, range(0, 175)), # temp
        train_dataset,
        processor,
        use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
        frame_sampling_ratio=data_args.frame_sampling_ratio,
        context_size_limit=data_args.context_size_limit,
    )
    smolvlm_train_time = time.time() - smolvlm_train_start
    logger.info(f"SmolVLM train dataset conversion took {smolvlm_train_time:.2f}s")

    logger.info("Initializing SmolVLM eval dataset...")
    smolvlm_eval_dataset = None
    if eval_datasets:
        smolvlm_eval_start = time.time()
        eval_dataset = list(eval_datasets.values())[0]  # Use first eval dataset
        smolvlm_eval_dataset = ProAssistSmolVLMDataset(
            # Subset(eval_dataset, range(0, 10)), # temp
            eval_dataset,
            processor,
            use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
            frame_sampling_ratio=data_args.frame_sampling_ratio,
            context_size_limit=data_args.context_size_limit,
        )
        smolvlm_eval_time = time.time() - smolvlm_eval_start
        logger.info(f"SmolVLM eval dataset conversion took {smolvlm_eval_time:.2f}s")

    if is_global_rank_zero():
        logger.info(f"Original train dataset size: {len(train_dataset)}")
        logger.info(f"Split train dataset size: {len(smolvlm_train_dataset)}")
        if smolvlm_eval_dataset:
            eval_dataset_size = len(list(eval_datasets.values())[0])
            logger.info(f"Original eval dataset size: {eval_dataset_size}")
            logger.info(f"Split eval dataset size: {len(smolvlm_eval_dataset)}")

    # Log memory after dataset creation
    log_memory_usage("After dataset creation", logger)

    # Initialize trainer
    logger.info("Initializing trainer...")
    logger.info(f"Train dataset size: {len(smolvlm_train_dataset) if smolvlm_train_dataset else 0}")
    logger.info(f"Eval dataset size: {len(smolvlm_eval_dataset) if smolvlm_eval_dataset else 0}")
    logger.info(f"Evaluation strategy: {training_args.eval_strategy}")
    logger.info(f"Eval steps: {training_args.eval_steps}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Per device train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Total steps per optimizer update: {training_args.gradient_accumulation_steps}")
    
    # Calculate when first eval should happen
    if training_args.eval_strategy == "steps" and training_args.eval_steps > 0:
        logger.info(f"First evaluation should happen at global step: {training_args.eval_steps}")
    
    # Determine W2T token ID based on configuration
    w2t_token_id = END_OF_UTTERANCE_TOKEN_ID if data_args.use_end_of_utterance_for_w2t else W2T_TOKEN_ID
    logger.info(f"Using W2T token ID: {w2t_token_id} ({'END_OF_UTTERANCE' if data_args.use_end_of_utterance_for_w2t else 'reserved_special_token_0'})")
    logger.info(f"W2T only mode: {data_args.w2t_only}")
    logger.info(f"No assistant mode: {data_args.no_assistant}")
    
    trainer_init_start = time.time()
    trainer = SmolVLMProAssistTrainer(
        processor=processor,
        w2t_frame_sampling_rate=data_args.w2t_frame_sampling_rate,
        w2t_token_id=w2t_token_id,
        w2t_only=data_args.w2t_only,
        no_assistant=data_args.no_assistant,
        model=model,
        args=training_args,
        train_dataset=smolvlm_train_dataset,
        eval_dataset=smolvlm_eval_dataset,
    )
    trainer_init_time = time.time() - trainer_init_start
    logger.info(f"Trainer initialization took {trainer_init_time:.2f}s")

    # Log memory after trainer initialization
    log_memory_usage("After trainer initialization", logger)
    log_gpu_utilization("After trainer initialization", logger)

    # logger.info("DONE CONVERTING!")
    # return # temp for converting proassist dataset to smolvlm format

    # Start training
    if is_global_rank_zero():
        log_memory_usage("Before training start", logger)
        log_gpu_utilization("Before training start", logger)
        logger.info("Starting training...")

    training_start_time = time.time()
    trainer.train()
    training_total_time = time.time() - training_start_time
    
    if is_global_rank_zero():
        logger.info(f"Training completed in {training_total_time:.2f}s ({training_total_time/3600:.2f}h)")
        log_memory_usage("After training", logger)
        log_gpu_utilization("After training", logger)


    # Save final model
    if training_args.local_rank == 0:
        if model_args.use_lora or model_args.use_qlora:
            # Save only LoRA adapter weights for easy combination later
            logger.info("Saving LoRA adapter weights...")
            model.save_pretrained(training_args.output_dir)
            
            # Save adapter metadata for tracking
            import json
            adapter_metadata = {
                "base_model": model_args.model_name_or_path,
                "train_datasets": data_args.train_datasets,
                "lora_r": lora_args.lora_r,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "target_modules": lora_args.lora_target_modules,
                "use_dora": lora_args.use_dora,
                "use_qlora": model_args.use_qlora,
                "training_args": {
                    "num_epochs": training_args.num_train_epochs,
                    "learning_rate": training_args.learning_rate,
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                }
            }
            
            metadata_path = os.path.join(training_args.output_dir, "adapter_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(adapter_metadata, f, indent=2)
            
            if is_global_rank_zero():
                logger.info(f"LoRA adapter saved to {training_args.output_dir}")
                logger.info(f"Adapter metadata saved to {metadata_path}")
                logger.info(f"Trained on dataset(s): {data_args.train_datasets}")
                logger.info("This adapter can be combined with other LoRA adapters using PEFT's weighted combination features")
        else:
            # Save full model if not using LoRA
            trainer.save_model()
            if is_global_rank_zero():
                logger.info(f"Full model saved to {training_args.output_dir}")
        
        # Save processor configuration
        processor.save_pretrained(training_args.output_dir)
        
        if is_global_rank_zero():
            logger.info(f"Processor saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
