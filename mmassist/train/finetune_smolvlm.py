"""
Fine-tune SmolVLM-Instruct on ProAssist dataset for streaming video assistance.

This script adapts the SmolVLM fine-tuning approach to work with ProAssist's 
multi-modal conversation format with temporal video frames.
"""

import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Tuple
import transformers
from transformers import (
    AutoProcessor, 
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig

# ProAssist imports
from mmassist.configs import parse_args
from mmassist.data import build_train_dataset, build_eval_datasets
from mmassist.data.utils import tensor_to_pil_images
from mmassist.train.utils import is_global_rank_zero


@dataclass
class SmolVLMModelArguments:
    model_name_or_path: Optional[str] = field(
        default="HuggingFaceTB/SmolVLM-Instruct",
        metadata={"help": "Path to pretrained SmolVLM model"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA (4-bit quantization)"}
    )
    freeze_vision: bool = field(
        default=False,
        metadata={"help": "Whether to freeze vision encoder during training"}
    )


@dataclass
class SmolVLMDataArguments:
    train_datasets: str = field(
        default="wtag/dialog-klg-sum_train_L2048_I1",
        metadata={"help": "Training datasets to use"}
    )
    eval_datasets: Optional[str] = field(
        default="wtag/dialog-klg-sum_val_L2048_I1",
        metadata={"help": "Evaluation datasets to use"}
    )
    data_root_dir: str = field(
        default="/projects/beto/swong2/proassist_data/processed_data",
        metadata={"help": "Root directory for ProAssist data"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for input"}
    )
    use_4_1_aspect_ratio: bool = field(
        default=True,
        metadata={"help": "Whether to use 4:1 aspect ratio for optimal SmolVLM encoding"}
    )
    frames_per_sample: int = field(
        default=5,
        metadata={"help": "Maximum number of frames to include per sample"}
    )
    frame_sampling_ratio: float = field(
        default=0.3,
        metadata={"help": "Ratio of frames to sample from frame ranges"}
    )
    context_size_limit: int = field(
        default=6000,
        metadata={"help": "Context size limit in tokens before splitting samples (leave room below 8k)"}
    )


@dataclass
class SmolVLMTrainingArguments(TrainingArguments):
    output_dir: str = field(default="/work/nvme/beto/swong2/smolvlm_proassist_finetune")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=25)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    optim: str = field(default="paged_adamw_8bit")
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
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


class ProAssistSmolVLMDataset:
    """Dataset adapter for ProAssist data to SmolVLM format."""
    
    def __init__(
        self,
        proassist_dataset,
        processor,
        max_seq_length: int = 2048,
        use_4_1_aspect_ratio: bool = True,
        frames_per_sample: int = 5,
        frame_sampling_ratio: float = 0.3,
        context_size_limit: int = 6000  # Leave room for 1-2 turns below 8k
    ):
        self.proassist_dataset = proassist_dataset
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.use_4_1_aspect_ratio = use_4_1_aspect_ratio
        self.frames_per_sample = frames_per_sample
        self.frame_sampling_ratio = frame_sampling_ratio
        self.context_size_limit = context_size_limit
        
        # Get image token ID for masking
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        
        # Preprocess and split long samples
        self.split_samples = []
        self._preprocess_and_split_samples()
    
    def __len__(self):
        return len(self.split_samples)
    
    def __getitem__(self, idx):
        return self.convert_proassist_to_smolvlm(self.split_samples[idx])
    
    def _preprocess_and_split_samples(self):
        """Preprocess samples to handle task knowledge and split long samples."""
        for sample in self.proassist_dataset:
            # Step 1: Fix task knowledge in system messages
            processed_sample = self._fix_task_knowledge(sample)
            
            # Step 2: Split long conversations into smaller samples
            split_samples = self._split_long_sample(processed_sample)
            self.split_samples.extend(split_samples)
    
    def _fix_task_knowledge(self, sample):
        """Fix task knowledge placement in system messages."""
        conversation = sample["conversation"].copy()
        task_knowledge = f"Task knowledge: {sample['metadata']['knowledge']}"
        
        # Find and process system messages
        first_system_idx = None
        second_system_idx = None
        
        for i, turn in enumerate(conversation):
            if turn["role"] == "system":
                if first_system_idx is None:
                    first_system_idx = i
                else:
                    second_system_idx = i
                    break
        
        # Remove second system turn if it contains "Task knowledge: "
        if second_system_idx is not None and "Task knowledge: " in conversation[second_system_idx]["content"]:
            conversation.pop(second_system_idx)
        
        # Add task knowledge to first system turn if not already present
        first_system_content = conversation[first_system_idx]["content"]
        if "Task knowledge: " not in first_system_content:
            conversation[first_system_idx]["content"] = f"{first_system_content}. {task_knowledge}"
        
        # Create updated sample
        updated_sample = sample.copy()
        updated_sample["conversation"] = conversation
        return updated_sample
    
    def _split_long_sample(self, sample):
        """Split a long sample into multiple smaller samples."""
        conversation = sample["conversation"]
        images = sample.get("images", [])
        
        # Extract assistant instruction from first system message
        assistant_instruction = self._extract_assistant_instruction(conversation)
        
        split_samples = []
        current_messages = []
        last_progress_summary = ""
        current_images = []
        
        i = 0
        while i < len(conversation):
            turn = conversation[i]
            
            # Temporarily add this turn to check token count
            temp_messages = current_messages.copy()
            temp_images = current_images.copy()
            
            # Process the turn and add to temp messages
            if turn["role"] == "system":
                temp_messages.append(turn)
            elif turn["role"] == "assistant":
                temp_messages.append(turn)
                # Save progress summary if available
                if "progress" in turn:
                    last_progress_summary = turn["progress"]
            elif turn["role"] == "user":
                temp_messages.append(turn)
            elif turn["role"] == "frames":
                # Sample frames and add as user message
                frame_content, sampled_images = self._process_frames_turn_with_images(turn, sample, images)
                if frame_content:
                    # Append to existing user message or create new one
                    if temp_messages and temp_messages[-1]["role"] == "user":
                        temp_messages[-1]["content"] += f"\n{frame_content}"
                    else:
                        temp_messages.append({"role": "user", "content": frame_content})
                    temp_images.extend(sampled_images)
            
            # Check if this would exceed our token limit
            if temp_messages:
                token_count = self._get_exact_token_count(temp_messages, temp_images)
                
                if token_count > self.context_size_limit and current_messages:
                    # Create a sample from current messages (before adding this turn)
                    split_sample = self._create_split_sample(
                        sample, current_messages, last_progress_summary, assistant_instruction
                    )
                    if split_sample:
                        split_sample["images"] = current_images
                        split_samples.append(split_sample)
                    
                    # Reset for next sample
                    current_messages = []
                    current_images = []
                    
                    # Start new sample with updated system message
                    if last_progress_summary:
                        system_msg = self._create_system_message(
                            assistant_instruction, last_progress_summary, sample["metadata"]["knowledge"]
                        )
                        current_messages.append({"role": "system", "content": system_msg})
                    
                    # Don't increment i - process this turn in the new sample
                    continue
            
            # Add the turn to current messages (commit the temp changes)
            current_messages = temp_messages
            current_images = temp_images
            i += 1
        
        # Add remaining messages as a final sample
        if current_messages:
            split_sample = self._create_split_sample(
                sample, current_messages, last_progress_summary, assistant_instruction
            )
            if split_sample:
                split_sample["images"] = current_images
                split_samples.append(split_sample)
        
        return split_samples if split_samples else [sample]  # Return original if no splitting occurred
            
    def _extract_assistant_instruction(self, conversation):
        """Extract assistant instruction from first system message."""
        for turn in conversation:
            if turn["role"] == "system":
                content = turn["content"]
                # Remove progress summary part
                if "The time elapsed since" in content:
                    content = content.split("The time elapsed since")[0].strip()
                return content
        
        # default
        return "You are a helpful and proactive assistant. Always be ready to assist and provide useful information ahead of time."
    
    def _get_exact_token_count(self, messages, images=None):
        """Get exact token count using SmolVLM processor."""
        try:
            # Convert messages to the format expected by apply_chat_template
            formatted_messages = []
            for msg in messages:
                if msg["role"] in ["system", "user", "assistant"]:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
            
            if not formatted_messages:
                return 0
            
            # Apply chat template and tokenize
            text = self.processor.apply_chat_template(
                formatted_messages,
                add_generation_prompt=False,
                tokenize=False
            )
            
            # Tokenize to get exact count
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            
            num_tokens = inputs['input_ids'].shape[1]
            
            # Add estimated tokens for images
            if images:
                # SmolVLM typically uses around 64 tokens per image
                num_tokens += len(images) * 64
            
            return num_tokens
            
        except Exception as e:
            # Fallback to character-based estimation if processor fails
            total_chars = sum(len(msg["content"]) for msg in messages if "content" in msg)
            return total_chars // 4  # Rough token estimation
    
    def _process_frames_turn_with_images(self, turn, sample, images):
        """Process frames turn and return both text content and actual images."""
        start = turn["start"] - sample["start_frame_idx"]
        end = turn["end"] - sample["start_frame_idx"]
        
        num_frames = max(0, min(end, len(images)) - max(0, start))
        if num_frames == 0:
            return "", []
        
        # Sample frames based on sampling ratio
        sampled_frame_count = max(1, int(num_frames * self.frame_sampling_ratio))
        sampled_frame_count = min(sampled_frame_count, self.frames_per_sample)
        
        if sampled_frame_count > 0:
            step = max(1, num_frames // sampled_frame_count)
            frame_indices = list(range(start, min(end, len(images)), step))[:sampled_frame_count]
            
            # Get actual image tensors
            sampled_images = []
            for k in frame_indices:
                if k < len(images):
                    sampled_images.append(images[k])
            
            frame_content = f"[Frames from {start} to {end}, sampled {len(sampled_images)} frames]"
            return frame_content, sampled_images
        
        return "", []
    
    def _process_frames_turn(self, turn, sample, images):
        """Process frames turn and return text description."""
        frame_content, _ = self._process_frames_turn_with_images(turn, sample, images)
        return frame_content
    
    def _create_system_message(self, assistant_instruction, progress_summary, knowledge):
        """Create system message with instruction, progress, and knowledge."""
        return f"{assistant_instruction}\n\n{progress_summary}\n\nTask knowledge: {knowledge}"
    
    def _create_split_sample(self, original_sample, messages, progress_summary, assistant_instruction):
        """Create a split sample from messages."""
        if not messages:
            return None
        
        # Remove latest turns that are not assistant turns
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()
        
        if not messages:
            return None
        
        # Add summary prompt before the last assistant turn if we have progress
        if len(messages) > 1 and progress_summary and messages[-1]["role"] == "assistant":
            summary_prompt = {"role": "system", "content": "Please summarize the progress."}
            messages.insert(-1, summary_prompt)
            
            # Set assistant content to progress summary
            messages[-1]["content"] = progress_summary
        
        # Create new sample
        split_sample = original_sample.copy()
        split_sample["conversation"] = messages
        return split_sample
    
    def resize_image_for_optimal_encoding(self, image):
        """Resize image to 4:1 aspect ratio for optimal SmolVLM encoding."""
        if not self.use_4_1_aspect_ratio:
            return image
        
        # Calculate target dimensions maintaining 4:1 ratio
        target_width = 384
        target_height = target_width // 4  # 4:1 ratio
        
        return image.resize((target_width, target_height))
    
    def convert_proassist_to_smolvlm(self, sample):
        """Convert preprocessed ProAssist sample to SmolVLM chat format."""
        conversation = sample["conversation"]
        images = sample.get("images", [])
        
        # Convert to SmolVLM chat format
        messages = []
        current_user_content = []
        frames_added = 0
        selected_images = []
        
        for turn in conversation:
            if turn["role"] == "system":
                # Add system message as user content
                if current_user_content:
                    current_user_content.append({"type": "text", "text": turn["content"]})
                else:
                    current_user_content = [{"type": "text", "text": turn["content"]}]
                    
            elif turn["role"] == "user":
                # Add user message to current content
                if current_user_content:
                    current_user_content.append({"type": "text", "text": turn["content"]})
                else:
                    current_user_content = [{"type": "text", "text": turn["content"]}]
                    
            elif turn["role"] == "frames":
                # Process frames in this segment
                start = turn["start"] - sample["start_frame_idx"]
                end = turn["end"] - sample["start_frame_idx"]
                
                # Sample frames based on sampling ratio
                num_frames = max(0, min(end, len(images)) - max(0, start))
                if num_frames > 0:
                    sampled_frames = max(1, int(num_frames * self.frame_sampling_ratio))
                    sampled_frames = min(sampled_frames, self.frames_per_sample - frames_added)
                    
                    # Get frame indices
                    if sampled_frames > 0:
                        step = max(1, num_frames // sampled_frames)
                        frame_indices = list(range(start, min(end, len(images)), step))[:sampled_frames]
                        
                        for k in frame_indices:
                            if frames_added >= self.frames_per_sample:
                                break
                                
                            # Convert tensor to PIL image
                            if k < len(images):
                                pt_img = images[k:k+1]
                                pil_img = tensor_to_pil_images(pt_img)[0]
                                pil_img = self.resize_image_for_optimal_encoding(pil_img)
                                
                                if not current_user_content:
                                    current_user_content = []
                                current_user_content.append({"type": "image", "image": pil_img})
                                selected_images.append(pil_img)
                                frames_added += 1
                
            elif turn["role"] == "assistant":
                # Add accumulated user content
                if current_user_content:
                    messages.append({
                        "role": "user",
                        "content": current_user_content
                    })
                    current_user_content = []
                
                # Add assistant response
                messages.append({
                    "role": "assistant", 
                    "content": [{"type": "text", "text": turn["content"]}]
                })
        
        # Add any remaining user content
        if current_user_content:
            messages.append({
                "role": "user",
                "content": current_user_content
            })
        
        # Skip samples without messages or images
        if not messages:
            return None
            
        return {
            "messages": messages,
            "images": selected_images,
            "sample_metadata": {
                "sample_idx": sample.get("sample_idx", -1),
                "video_uid": sample.get("video_uid", "unknown"),
                "num_frames": len(selected_images)
            }
        }


def collate_fn(examples, processor):
    """Collate function for SmolVLM training."""
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
            messages, 
            add_generation_prompt=False,
            tokenize=False
        )
        texts.append(text.strip())
        
        # Collect all images from this conversation
        images = example["images"]
        all_images.append(images)
    
    # Process batch
    batch = processor(
        text=texts, 
        images=all_images, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Create labels for training
    labels = batch["input_ids"].clone()
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens (model shouldn't predict image tokens)
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    
    return batch


class SmolVLMProAssistTrainer(Trainer):
    """Custom trainer for SmolVLM ProAssist fine-tuning."""
    
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
    
    def get_train_dataloader(self):
        """Override to use custom collate function."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        from functools import partial
        from torch.utils.data import DataLoader
        
        collate_fn_with_processor = partial(collate_fn, processor=self.processor)
        
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
        
        collate_fn_with_processor = partial(collate_fn, processor=self.processor)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_processor,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def setup_model_and_processor(model_args: SmolVLMModelArguments, lora_args: LoraArguments):
    """Setup SmolVLM model and processor with optional LoRA."""
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Setup quantization if using QLoRA
    quantization_config = None
    if model_args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Freeze vision encoder if requested
    if model_args.freeze_vision:
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
        if is_global_rank_zero():
            print("Vision encoder frozen")
    
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
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, processor


def main():
    parser = HfArgumentParser((
        SmolVLMModelArguments, 
        SmolVLMDataArguments, 
        SmolVLMTrainingArguments,
        LoraArguments
    ))
    
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    if is_global_rank_zero():
        print("="*50)
        print("SmolVLM ProAssist Fine-tuning")
        print("="*50)
        print(f"Model: {model_args.model_name_or_path}")
        print(f"Use LoRA: {model_args.use_lora}")
        print(f"Use QLoRA: {model_args.use_qlora}")
        print(f"Freeze Vision: {model_args.freeze_vision}")
        print(f"Train datasets: {data_args.train_datasets}")
        print(f"Eval datasets: {data_args.eval_datasets}")
        print(f"Max sequence length: {data_args.max_seq_length}")
        print(f"4:1 aspect ratio: {data_args.use_4_1_aspect_ratio}")
        print(f"Frames per sample: {data_args.frames_per_sample}")
        print(f"Frame sampling ratio: {data_args.frame_sampling_ratio}")
        print(f"Context size limit: {data_args.context_size_limit}")
    
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
        print("\nLoading ProAssist datasets...")
    
    train_dataset = build_train_dataset(**all_args_dict)
    eval_datasets = build_eval_datasets(**all_args_dict) if data_args.eval_datasets else {}
    
    # Convert to SmolVLM format
    smolvlm_train_dataset = ProAssistSmolVLMDataset(
        train_dataset,
        processor,
        max_seq_length=data_args.max_seq_length,
        use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
        frames_per_sample=data_args.frames_per_sample,
        frame_sampling_ratio=data_args.frame_sampling_ratio,
        context_size_limit=data_args.context_size_limit
    )
    
    smolvlm_eval_dataset = None
    if eval_datasets:
        eval_dataset = list(eval_datasets.values())[0]  # Use first eval dataset
        smolvlm_eval_dataset = ProAssistSmolVLMDataset(
            eval_dataset,
            processor,
            max_seq_length=data_args.max_seq_length,
            use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
            frames_per_sample=data_args.frames_per_sample,
            frame_sampling_ratio=data_args.frame_sampling_ratio,
            context_size_limit=data_args.context_size_limit
        )
    
    if is_global_rank_zero():
        print(f"\nOriginal train dataset size: {len(train_dataset)}")
        print(f"Split train dataset size: {len(smolvlm_train_dataset)}")
        if smolvlm_eval_dataset:
            eval_dataset_size = len(list(eval_datasets.values())[0])
            print(f"Original eval dataset size: {eval_dataset_size}")
            print(f"Split eval dataset size: {len(smolvlm_eval_dataset)}")
    
    # Initialize trainer
    trainer = SmolVLMProAssistTrainer(
        processor=processor,
        model=model,
        args=training_args,
        train_dataset=smolvlm_train_dataset,
        eval_dataset=smolvlm_eval_dataset,
    )
    
    # Start training
    if is_global_rank_zero():
        print("\nStarting training...")
    
    trainer.train()
    
    # Save final model
    if training_args.local_rank == 0:
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        if is_global_rank_zero():
            print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
