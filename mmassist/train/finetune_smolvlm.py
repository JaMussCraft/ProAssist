"""
Fine-tune SmolVLM-Instruct on ProAssist dataset for streaming video assistance.

This script adapts the SmolVLM fine-tuning approach to work with ProAssist's
multi-modal conversation format with temporal video frames.
"""

import os
import torch
import logging
import pickle
import numpy as np
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
    logging_steps: int = field(default=25)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    optim: str = field(default="paged_adamw_8bit")
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True) # saves memory; recompute gradients during backward pass
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="tensorboard")


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: str = field(
        default="down_proj,o_proj,k_proj,q_proj,gate_proj,up_proj,v_proj"
    )
    use_dora: bool = field(default=False)


def collate_fn(examples, processor, w2t_frame_sampling_rate=0.3):
    """Enhanced collate function for SmolVLM training with w2t token support."""
    # Filter out None examples
    examples = [ex for ex in examples if ex is not None]

    if not examples:
        return None

    texts = []
    all_images = []

    for example in examples:
        messages = example["messages"]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False
        )
        texts.append(text.strip())

        # Collect all images from this conversation
        images = example["images"]


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

    # Process batch
    batch = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
        truncation=True, # the samples shouldn't exceed max_length (probably...)
        max_length=8192,
    )

    # Create labels with advanced w2t masking
    labels = torch.full_like(batch["input_ids"], -100, dtype=torch.long)

    # Process each sample in the batch
    for i, input_ids in enumerate(batch["input_ids"]):
        # Get learning ranges for this sample
        learn_ranges = get_smolvlm_learn_ranges(
            input_ids, frame_sampling_rate=w2t_frame_sampling_rate
        )
        
        # Apply masking based on range types
        for start_idx, end_idx, label_type in learn_ranges:
            if label_type == 'assistant':
                # Learn from actual assistant tokens
                labels[i, start_idx:end_idx] = input_ids[start_idx:end_idx]
            elif label_type == 'w2t':
                # Learn to predict w2t token at frame decision point
                labels[i, start_idx] = W2T_TOKEN_ID
            elif label_type == 'talk':
                # Learn to predict the actual next token (positive frame)
                if end_idx < len(input_ids):
                    labels[i, start_idx] = input_ids[end_idx]

    # Mask padding tokens
    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens (model shouldn't predict image tokens)
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    labels[labels == image_token_id] = -100

    batch["labels"] = labels


    return batch


class SmolVLMProAssistTrainer(Trainer):
    """Custom trainer for SmolVLM ProAssist fine-tuning."""

    def __init__(self, processor, w2t_frame_sampling_rate=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.w2t_frame_sampling_rate = w2t_frame_sampling_rate

    def get_train_dataloader(self):
        """Override to use custom collate function."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        from functools import partial
        from torch.utils.data import DataLoader

        collate_fn_with_processor = partial(
            collate_fn, 
            processor=self.processor,
            w2t_frame_sampling_rate=self.w2t_frame_sampling_rate
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_processor,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

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
            w2t_frame_sampling_rate=self.w2t_frame_sampling_rate
        )

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_processor,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate_speaking_decisions(self, eval_dataset=None):
        """
        Evaluate model performance on speaking decision prediction (w2t token prediction).
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, F1
        """
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        logger = logging.getLogger(__name__)
        logger.info("Starting speaking decision evaluation...")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
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
                    
                    # Extract speaking decision positions and predictions
                    for start_idx, end_idx, label_type in learn_ranges:
                        if label_type in ['w2t', 'talk']:
                            if start_idx < logits.shape[1]:
                                # Get prediction at decision point
                                position_logits = logits[i, start_idx]  # [vocab_size]
                                
                                # Get probabilities for W2T vs other tokens
                                probs = F.softmax(position_logits, dim=-1)
                                w2t_prob = probs[W2T_TOKEN_ID].item()
                                
                                # Prediction: speak if w2t_prob < 0.5, else wait
                                predicted_speak = w2t_prob < 0.5
                                
                                # Ground truth: speak if label_type is 'talk'
                                true_speak = (label_type == 'talk')
                                
                                all_predictions.append(predicted_speak)
                                all_targets.append(true_speak)
                                
                                if label_type == 'w2t':
                                    total_w2t_positions += 1
                                else:  # 'talk'
                                    total_talk_positions += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(eval_dataloader)}")
        
        if not all_predictions:
            logger.warning("No speaking decision positions found in evaluation data")
            return {}
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'speaking_decision_accuracy': accuracy,
            'speaking_decision_precision': precision,
            'speaking_decision_recall': recall,
            'speaking_decision_f1': f1,
            'speaking_decision_specificity': specificity,
            'total_decision_points': len(all_predictions),
            'total_w2t_positions': total_w2t_positions,
            'total_talk_positions': total_talk_positions,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        logger.info("Speaking Decision Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  Total decision points: {len(all_predictions)}")
        logger.info(f"  W2T positions: {total_w2t_positions}")
        logger.info(f"  Talk positions: {total_talk_positions}")
        logger.info(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
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
    logger = logging.getLogger(__name__)
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}") # make sure this lists all gpus

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
        logger.info(f"Context size limit: {data_args.context_size_limit}")

    # Setup model and processor
    model, processor = setup_model_and_processor(model_args, lora_args)

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

    train_dataset = build_train_dataset(**all_args_dict)
    eval_datasets = (
        build_eval_datasets(**all_args_dict) if data_args.eval_datasets else {}
    )

    from torch.utils.data import Subset

    # Convert to SmolVLM format
    logger.info("Initializing SmolVLM train dataset...")
    smolvlm_train_dataset = ProAssistSmolVLMDataset(
        # Subset(train_dataset, range(7980, len(train_dataset))), # temp
        train_dataset,
        processor,
        use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
        frame_sampling_ratio=data_args.frame_sampling_ratio,
        context_size_limit=data_args.context_size_limit,
    )

    logger.info("Initializing SmolVLM eval dataset...")
    smolvlm_eval_dataset = None
    if eval_datasets:
        eval_dataset = list(eval_datasets.values())[0]  # Use first eval dataset
        smolvlm_eval_dataset = ProAssistSmolVLMDataset(
            # Subset(eval_dataset, range(0, 10)), # temp
            eval_dataset,
            processor,
            use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
            frame_sampling_ratio=data_args.frame_sampling_ratio,
            context_size_limit=data_args.context_size_limit,
        )

    if is_global_rank_zero():
        logger.info(f"Original train dataset size: {len(train_dataset)}")
        logger.info(f"Split train dataset size: {len(smolvlm_train_dataset)}")
        if smolvlm_eval_dataset:
            eval_dataset_size = len(list(eval_datasets.values())[0])
            logger.info(f"Original eval dataset size: {eval_dataset_size}")
            logger.info(f"Split eval dataset size: {len(smolvlm_eval_dataset)}")

    # Initialize trainer
    trainer = SmolVLMProAssistTrainer(
        processor=processor,
        w2t_frame_sampling_rate=data_args.w2t_frame_sampling_rate,
        model=model,
        args=training_args,
        train_dataset=smolvlm_train_dataset,
        eval_dataset=smolvlm_eval_dataset,
    )


    logger.info("DONE CONVERTING!")
    return # temp for converting proassist dataset to smolvlm format

    # Start training
    if is_global_rank_zero():
        logger.info("Starting training...")

    trainer.train()

    # Evaluate speaking decisions after training
    if smolvlm_eval_dataset and is_global_rank_zero():
        logger.info("Evaluating speaking decisions...")
        eval_metrics = trainer.evaluate_speaking_decisions(smolvlm_eval_dataset)
        
        # Log metrics to tensorboard if available
        if hasattr(trainer, 'log'):
            trainer.log(eval_metrics)

    # Save final model
    if training_args.local_rank == 0:
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)

        if is_global_rank_zero():
            logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
