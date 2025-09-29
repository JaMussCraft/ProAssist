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
    HfArgumentParser,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig

# ProAssist imports
from mmassist.configs import parse_args
from mmassist.data import build_train_dataset, build_eval_datasets
from mmassist.data.utils import tensor_to_pil_images
from mmassist.train.utils import is_global_rank_zero

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
    max_seq_length: int = field(
        default=8192, metadata={"help": "Maximum sequence length for input"}
    )
    use_4_1_aspect_ratio: bool = field(
        default=True,
        metadata={
            "help": "Whether to use 4:1 aspect ratio for optimal SmolVLM encoding"
        },
    )
    frame_sampling_ratio: float = field(
        default=0.3, metadata={"help": "Ratio of frames to sample from frame ranges"}
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
    """Dataset adapter for ProAssist data to SmolVLM format.

    The proassist samples are both split and converted to SmolVLM format at the same time.
    """

    def __init__(
        self,
        proassist_dataset,
        processor,
        max_seq_length: int = 8192,
        use_4_1_aspect_ratio: bool = True,
        frame_sampling_ratio: float = 0.3,
        context_size_limit: int = 7500,  # Leave room for 1-2 turns below 8k
    ):
        self.proassist_dataset = proassist_dataset
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.use_4_1_aspect_ratio = use_4_1_aspect_ratio
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
        return self.split_samples[idx]

    def _preprocess_and_split_samples(self):
        """Preprocess samples to handle task knowledge and split long samples."""
        for sample in self.proassist_dataset:
            # Step 1: Fix task knowledge in system messages
            processed_sample = self._fix_task_knowledge(sample)

            # Step 2: Split and convert samples into smolVLM format
            self._split_and_convert_proassist_to_smolvlm(processed_sample)

            print(f"One proassist sample turned into {len(self.split_samples)} smolVLM samples")
            break

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
        if (
            second_system_idx is not None
            and "Task knowledge: " in conversation[second_system_idx]["content"]
        ):
            conversation.pop(second_system_idx)

        # Add task knowledge to first system turn if not already present
        first_system_content = conversation[first_system_idx]["content"]
        if "Task knowledge: " not in first_system_content:
            conversation[first_system_idx][
                "content"
            ] = f"{first_system_content} {task_knowledge}"

        # Create updated sample
        updated_sample = sample.copy()
        updated_sample["conversation"] = conversation
        return updated_sample

    def _split_and_convert_proassist_to_smolvlm(self, sample):
        """
        Split a long proassist sample into multiple smaller samples
        while converting to smolVLM format.
        """
        conversation = sample["conversation"]
        images = sample.get("images", [])

        # Extract assistant instruction from first system message
        assistant_instruction = self._extract_assistant_instruction(conversation)

        current_messages = []
        last_progress_summary = ""
        current_images = []

        i = 0
        while i < len(conversation):
            turn = conversation[i]

            # Process the turn
            if turn["role"] == "system":
                current_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": turn["content"]}],
                    }
                )

            elif turn["role"] == "assistant":
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn["content"]}],
                    }
                )

                # Save progress summary if available
                if "progress" in turn:
                    last_progress_summary = turn["progress"]

            elif turn["role"] == "user":
                if (
                    current_messages and current_messages[-1]["role"] == "user"
                ):  # latest turn is user
                    current_messages[-1]["content"].append(
                        {"type": "text", "text": turn["content"]}
                    )

                else:
                    current_messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": turn["content"]}],
                        }
                    )

            elif turn["role"] == "frames":
                # Sample frames and add as user message
                start = turn["start"] - sample["start_frame_idx"]
                end = turn["end"] - sample["start_frame_idx"]

                num_frames = max(0, min(end, len(images)) - max(0, start))

                # Sample frames based on sampling ratio
                sampled_frame_count = max(
                    1, int(num_frames * self.frame_sampling_ratio)
                )

                if sampled_frame_count > 0:
                    step = max(1, num_frames // sampled_frame_count)
                    frame_indices = list(range(start, min(end, len(images)), step))[
                        :sampled_frame_count
                    ]

                    # Get actual image tensors
                    sampled_images = []
                    for k in frame_indices:
                        if k < len(images):
                            sampled_images.append(images[k : k + 1])

                    print(f"Sampled {len(sampled_images)} frames from {start} to {end}")

                    sampled_pil_images = []
                    for pt_img in sampled_images:
                        pil_img = tensor_to_pil_images(pt_img)[0]
                        pil_img = self.resize_image_for_optimal_encoding(pil_img)

                        if (
                            current_messages and current_messages[-1]["role"] == "user"
                        ):  # latest turn is user
                            current_messages[-1]["content"].append({"type": "image"})

                        else:
                            current_messages.append(
                                {"role": "user", "content": [{"type": "image"}]}
                            )

                        current_images.append(pil_img)

            print("YIPPIE: ")
            print("New turn: ", turn)
            # print("New messages: ", current_messages)
            print(f"len(current_images): {len(current_images)}")

            # Check if this would exceed our token limit using smolVLM processor
            prompt = self.processor.apply_chat_template(
                current_messages, add_generation_prompt=False
            )
            inputs = self.processor(
                text=prompt,
                images=current_images if current_images else None,
                return_tensors="pt",
            )

            # Count tokens
            token_count = inputs["input_ids"].shape[1]
            print(f"Current messages' total token count: {token_count}")
            num_image_tokens = (inputs["input_ids"] == self.image_token_id).sum().item()
            print(f"Current messages' image token count: {num_image_tokens}")

            # add split sample when context_size is reached
            if current_messages and token_count > self.context_size_limit:
                # add a system role summary prompt followed by an assistant progress summary turn
                current_messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "Please summarize the progress."}
                        ],
                    }
                )

                current_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": last_progress_summary}],
                    }
                )

                # Create new sample
                split_sample = {
                    "messages": current_messages,
                    "images": current_images,
                    "sample_metadata": {
                        "sample_idx": sample.get("sample_idx", -1),
                        "video_uid": sample.get("video_uid", "unknown"),
                        "num_frames": len(current_images),
                    },
                }

                self.split_samples.append(split_sample)

                # Reset for next sample
                current_messages = []
                current_images = []

                # Start new sample with updated system message
                if last_progress_summary:
                    system_msg = self._create_system_message(
                        assistant_instruction,
                        last_progress_summary,
                        sample["metadata"]["knowledge"],
                    )
                    current_messages.append({"role": "system", "content": system_msg})

            i += 1

        # Add unfull/remaining messages as a final sample
        if current_messages:
            split_sample = split_sample = {
                "messages": current_messages,
                "images": current_images,
                "sample_metadata": {
                    "sample_idx": sample.get("sample_idx", -1),
                    "video_uid": sample.get("video_uid", "unknown"),
                    "num_frames": len(current_images),
                },
            }

            self.split_samples.append(split_sample)

    def _extract_assistant_instruction(self, conversation):
        """Extract assistant instruction from first system message."""
        for turn in conversation:
            if turn["role"] == "system":
                content = turn["content"]

                # Remove progress summary part
                if "The time elapsed since" in content:
                    content = content.split("The time elapsed since")[0].strip()

                # Remove task knowledge part
                if "Task knowledge: " in content:
                    content = content.split("Task knowledge: ")[0].strip()

                return content

        # default
        return "You are a helpful and proactive assistant. Always be ready to assist and provide useful information ahead of time."

    def _process_frames_turn_with_images(self, turn, sample, images):
        """Process frames turn and return both text content and actual images."""
        start = turn["start"] - sample["start_frame_idx"]
        end = turn["end"] - sample["start_frame_idx"]

        num_frames = max(0, min(end, len(images)) - max(0, start))
        if num_frames == 0:
            return "", []

        # Sample frames based on sampling ratio
        sampled_frame_count = max(1, int(num_frames * self.frame_sampling_ratio))

        if sampled_frame_count > 0:
            step = max(1, num_frames // sampled_frame_count)
            frame_indices = list(range(start, min(end, len(images)), step))[
                :sampled_frame_count
            ]

            # Get actual image tensors
            sampled_images = []
            for k in frame_indices:
                if k < len(images):
                    sampled_images.append(images[k])

            frame_content = (
                f"[Frames from {start} to {end}, sampled {len(sampled_images)} frames]"
            )
            return frame_content, sampled_images

        return "", []

    def _create_system_message(
        self, assistant_instruction, progress_summary, knowledge
    ):
        """Create system message with instruction, progress, and knowledge."""
        return f"{assistant_instruction}\n\n{progress_summary}\n\nTask knowledge: {knowledge}"

    def resize_image_for_optimal_encoding(self, image):
        """Resize image to 4:1 aspect ratio for optimal SmolVLM encoding."""
        if not self.use_4_1_aspect_ratio:
            return image

        # Calculate target dimensions maintaining 4:1 ratio
        target_width = 384
        target_height = target_width // 4  # 4:1 ratio

        return image.resize((target_width, target_height))


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
            messages, add_generation_prompt=False, tokenize=False
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
        truncation=True, # the samples shouldn't exceed max_length (probably...)
        max_length=8192,
    )

    # Create labels with advanced w2t masking
    labels = torch.full_like(batch["input_ids"], -100, dtype=torch.long)
    print("THIS IS THE SHAPE: ", batch["input_ids"].shape)

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


def setup_model_and_processor(
    model_args: SmolVLMModelArguments, lora_args: LoraArguments
):
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
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(
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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if is_global_rank_zero():
        print("=" * 50)
        print("SmolVLM ProAssist Fine-tuning")
        print("=" * 50)
        print(f"Model: {model_args.model_name_or_path}")
        print(f"Use LoRA: {model_args.use_lora}")
        print(f"Use QLoRA: {model_args.use_qlora}")
        print(f"Freeze Vision: {model_args.freeze_vision}")
        print(f"Train datasets: {data_args.train_datasets}")
        print(f"Eval datasets: {data_args.eval_datasets}")
        print(f"Max sequence length: {data_args.max_seq_length}")
        print(f"4:1 aspect ratio: {data_args.use_4_1_aspect_ratio}")
        print(f"Frame sampling ratio: {data_args.frame_sampling_ratio}")
        print(f"W2T frame sampling rate: {data_args.w2t_frame_sampling_rate}")
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
    eval_datasets = (
        build_eval_datasets(**all_args_dict) if data_args.eval_datasets else {}
    )

    # Convert to SmolVLM format
    smolvlm_train_dataset = ProAssistSmolVLMDataset(
        train_dataset,
        processor,
        max_seq_length=data_args.max_seq_length,
        use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
        frame_sampling_ratio=data_args.frame_sampling_ratio,
        context_size_limit=data_args.context_size_limit,
    )

    smolvlm_eval_dataset = None
    if eval_datasets:
        eval_dataset = list(eval_datasets.values())[0]  # Use first eval dataset
        smolvlm_eval_dataset = ProAssistSmolVLMDataset(
            eval_dataset,
            processor,
            max_seq_length=data_args.max_seq_length,
            use_4_1_aspect_ratio=data_args.use_4_1_aspect_ratio,
            frame_sampling_ratio=data_args.frame_sampling_ratio,
            context_size_limit=data_args.context_size_limit,
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
        w2t_frame_sampling_rate=data_args.w2t_frame_sampling_rate,
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
