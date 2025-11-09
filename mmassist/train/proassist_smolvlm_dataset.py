import os
import logging
import pickle
from mmassist.data.utils import tensor_to_pil_images


class ProAssistSmolVLMDataset:
    """Dataset adapter for ProAssist data to SmolVLM format.

    The proassist samples are both split and converted to SmolVLM format at the same time.
    """

    def __init__(
        self,
        proassist_dataset,
        processor,
        use_4_1_aspect_ratio: bool = True,
        frame_sampling_ratio: float = 0.1, # proassist is 2 FPS; this will give 1 frame every 5s
        context_size_limit: int = 7500,  # Leave room for 1-2 turns below 8k
    ):
        self.proassist_dataset = proassist_dataset
        self.processor = processor
        self.use_4_1_aspect_ratio = use_4_1_aspect_ratio
        self.frame_sampling_ratio = frame_sampling_ratio
        self.context_size_limit = context_size_limit
        self.logger = logging.getLogger(__name__)

        # Get image token ID for masking
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        
        # Determine image tokens per frame based on model
        # Extract model name from processor's model config
        model_name = getattr(processor, 'model_name', None) or getattr(processor, '_model_name', None)
        if model_name is None and hasattr(processor, 'image_processor') and hasattr(processor.image_processor, '_name_or_path'):
            model_name = processor.image_processor._name_or_path
        if model_name is None and hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'name_or_path'):
            model_name = processor.tokenizer.name_or_path
        
        self.model_name = model_name or "unknown"
        
        # Set tokens per image and max images per sample based on model
        if "SmolVLM2-500M" in self.model_name or "500M" in self.model_name:
            self.tokens_per_image = 320
            self.max_images_per_sample = 22
        elif "SmolVLM2-2.2B" in self.model_name or "2.2B" in self.model_name or "2B" in self.model_name:
            self.tokens_per_image = 405
            self.max_images_per_sample = 18
        else:
            # Default to 320 for unknown models
            self.tokens_per_image = 320
            self.max_images_per_sample = 22
            self.logger.warning(f"Unknown model '{self.model_name}', defaulting to 320 tokens per image and 22 max images")
        
        # Adjust tokens per image and max images if not using 4:1 aspect ratio
        if not self.use_4_1_aspect_ratio:
            self.tokens_per_image = int(self.tokens_per_image * 3.4)
            self.max_images_per_sample = int(self.max_images_per_sample / 3.4)
        
        self.logger.info(f"Using {self.tokens_per_image} tokens per image and max {self.max_images_per_sample} images per sample for model: {self.model_name}")

        # Generate cache file path based on dataset and parameters
        self.cache_file_path = self._generate_cache_file_path()
        
        # Try to load existing processed data, otherwise preprocess and split
        if os.path.exists(self.cache_file_path):
            self.logger.info(f"Loading existing processed data from {self.cache_file_path}")
            self._load_split_samples()
        else:
            self.logger.info(f"Processing data from scratch and saving to {self.cache_file_path}")
            self.split_samples = []
            self._preprocess_and_split_samples()
            self._save_split_samples()
        
        self.logger.info(f"Dataset initialized with {len(self.split_samples)} split samples")

    def _generate_cache_file_path(self):
        """Generate cache file path based on dataset and parameters."""
        dataset_full = self.proassist_dataset[0]["dataset"]
        dataset_parts = dataset_full.split('/')
        
        dataset_name, samples = dataset_parts
        
        # Extract a clean model identifier from the model name
        model_id = "unknown"
        if "500M" in self.model_name:
            model_id = "500M"
        elif "2.2B" in self.model_name or "2B" in self.model_name:
            model_id = "2B"
        elif self.model_name != "unknown":
            # Use a simplified version of the model name
            model_id = self.model_name.split('/')[-1].replace('-', '_')
        
        # Create filename with all relevant parameters including model
        filename = f"smolvlm_processed_{samples}_model_{model_id}_4to1_{self.use_4_1_aspect_ratio}_sampling_{self.frame_sampling_ratio}_context_{self.context_size_limit}.pkl"
        
        # Create directory path
        cache_dir = f"/projects/beto/proassist_data/processed_data/{dataset_name}/prepared_smolvlm"
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, filename)
    
    def _load_split_samples(self):
        """Load split samples from cache file."""
        with open(self.cache_file_path, 'rb') as f:
            self.split_samples = pickle.load(f)
            # self.split_samples = self.split_samples[:250] # temp
        self.logger.info(f"Loaded {len(self.split_samples)} split samples from cache")
    
    def _save_split_samples(self):
        """Save split samples to cache file."""
        with open(self.cache_file_path, 'wb') as f:
            pickle.dump(self.split_samples, f)
        self.logger.info(f"Saved {len(self.split_samples)} split samples to {self.cache_file_path}")

    def __len__(self):
        return len(self.split_samples)

    def __getitem__(self, idx):
        """Get a processed sample with dynamically loaded images."""
        sample = self.split_samples[idx]
        
        # Create a copy to avoid modifying cached data
        result = {
            "messages": sample["messages"],
            "sample_metadata": sample["sample_metadata"]
        }
        
        # Dynamically load images from the original dataset
        if "image_references" in sample:
            result["images"] = self._load_images_from_references(sample["image_references"])
        else:
            result["images"] = []
        
        return result
    
    def _load_images_from_references(self, image_references):
        """Load PIL images from references to original dataset."""
        loaded_images = []
        for ref in image_references:
            sample_idx = ref["sample_idx"]
            frame_idx = ref["frame_idx"]
            
            # Get the original sample
            original_sample = self.proassist_dataset[sample_idx]
            images = original_sample.get("images", [])
            
            # Load the specific frame
            if frame_idx < len(images):
                pt_img = images[frame_idx:frame_idx + 1]
                pil_img = tensor_to_pil_images(pt_img)[0]
                pil_img = self.resize_image_for_optimal_encoding(pil_img)
                loaded_images.append(pil_img)
            else:
                self.logger.warning(f"Frame index {frame_idx} out of bounds for sample {sample_idx}")
        
        return loaded_images

    def _preprocess_and_split_samples(self):
        """Preprocess samples to handle task knowledge and split long samples."""

        i = 0
        for sample in self.proassist_dataset:
            # Step 1: Fix task knowledge in system messages
            processed_sample = self._fix_task_knowledge(sample)

            # Step 2: Split and convert samples into smolVLM format
            self._split_and_convert_proassist_to_smolvlm(processed_sample)

            self.logger.info(f"Processed sample {i+1}: now have {len(self.split_samples)} total split samples")

            i += 1
            # if i > 15: break # temp

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

    def _count_tokens_for_messages(self, messages, images=None):
        """
        Count tokens for a list of messages with optional images.
        Returns (total_tokens, image_tokens).
        """
        if not messages:
            return 0, 0
            
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        )
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        )
        
        total_tokens = inputs["input_ids"].shape[1]
        image_tokens = (inputs["input_ids"] == self.image_token_id).sum().item()
        return total_tokens, image_tokens

    def _count_tokens_for_single_message(self, message, images=None):
        """
        Count tokens for a single message with optional images.
        Returns (total_tokens, image_tokens).
        
        Uses a fixed token count per image for efficiency instead of
        actually processing images through the processor.
        Token count depends on the model:
        - 320 for SmolVLM2-500M-Video-Instruct
        - 405 for SmolVLM2-2.2B-Instruct
        """
        # For image-only messages, use fixed token count per image
        if images and len(images) > 0:
            # Check if this is an image-only message
            is_image_only = all(
                content.get("type") == "image" 
                for content in message.get("content", [])
            )
            
            if is_image_only:
                # Each image is encoded into a model-specific number of tokens
                num_images = len(images)
                image_tokens = num_images * self.tokens_per_image
                return image_tokens, image_tokens
        
        # For text or mixed messages, process normally
        return self._count_tokens_for_messages([message], images=None)

    def _split_and_convert_proassist_to_smolvlm(self, sample):
        """
        Split a long proassist sample into multiple smaller samples
        while converting to smolVLM format.
        """
        conversation = sample["conversation"]
        images = sample.get("images", [])
        sample_idx = sample.get("sample_idx", -1)

        # Extract assistant instruction from first system message
        assistant_instruction = self._extract_assistant_instruction(conversation)

        current_messages = []
        last_progress_summary = ""
        current_image_references = []
        
        # For incremental token counting optimization
        current_token_count = 0
        current_image_token_count = 0

        i = 0
        while i < len(conversation):
            turn = conversation[i]

            # Process the turn and update token counts incrementally
            new_tokens = 0
            new_image_tokens = 0
            
            if turn["role"] == "system":
                new_message = {
                    "role": "system",
                    "content": [{"type": "text", "text": turn["content"]}],
                }
                current_messages.append(new_message)
                
                # Count tokens for this message only
                msg_tokens, msg_img_tokens = self._count_tokens_for_single_message(new_message)
                new_tokens += msg_tokens
                new_image_tokens += msg_img_tokens

            elif turn["role"] == "assistant":
                new_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["content"]}],
                }
                current_messages.append(new_message)
                
                # Count tokens for this message only
                msg_tokens, msg_img_tokens = self._count_tokens_for_single_message(new_message)
                new_tokens += msg_tokens
                new_image_tokens += msg_img_tokens

                # Save progress summary if available
                if "progress" in turn:
                    last_progress_summary = turn["progress"]

            elif turn["role"] == "user":
                if (
                    current_messages and current_messages[-1]["role"] == "user"
                ):  # latest turn is user
                    # Adding text to existing user message
                    new_content = {"type": "text", "text": turn["content"]}
                    current_messages[-1]["content"].append(new_content)
                    
                    # Count tokens for just this new content
                    temp_message = {
                        "role": "user",
                        "content": [new_content]
                    }
                    msg_tokens, msg_img_tokens = self._count_tokens_for_single_message(temp_message)
                    new_tokens += msg_tokens
                    new_image_tokens += msg_img_tokens

                else:
                    # Creating new user message
                    new_message = {
                        "role": "user",
                        "content": [{"type": "text", "text": turn["content"]}],
                    }
                    current_messages.append(new_message)
                    
                    # Count tokens for this message only
                    msg_tokens, msg_img_tokens = self._count_tokens_for_single_message(new_message)
                    new_tokens += msg_tokens
                    new_image_tokens += msg_img_tokens

            elif turn["role"] == "frames":
                # Sample frames and add as user message
                start = turn["start"] - sample["start_frame_idx"]
                end = turn["end"] - sample["start_frame_idx"]

                num_frames = max(0, min(end, len(images)) - max(0, start))

                # Sample frames based on sampling ratio
                sampled_frame_count = min(max(
                    1, int(num_frames * self.frame_sampling_ratio)
                ), self.max_images_per_sample)

                if sampled_frame_count > 0:
                    step = max(1, num_frames // sampled_frame_count)
                    frame_indices = list(range(start, min(end, len(images)), step))[
                        :sampled_frame_count
                    ]

                    # Store references to frames instead of actual images
                    sampled_pil_images = []
                    new_image_content = []
                    for k in frame_indices:
                        if k < len(images):
                            # Add reference to the image
                            image_ref = {
                                "sample_idx": sample_idx,
                                "frame_idx": k
                            }
                            current_image_references.append(image_ref)
                            
                            # Load image temporarily for token counting
                            pt_img = images[k:k + 1]
                            pil_img = tensor_to_pil_images(pt_img)[0]
                            pil_img = self.resize_image_for_optimal_encoding(pil_img)
                            sampled_pil_images.append(pil_img)

                            if (
                                current_messages and current_messages[-1]["role"] == "user"
                            ):  # latest turn is user
                                current_messages[-1]["content"].append({"type": "image"})

                            else:
                                current_messages.append(
                                    {"role": "user", "content": [{"type": "image"}]}
                                )

                            new_image_content.append({"type": "image"})
                    
                    # Count tokens for the new images
                    if new_image_content:
                        temp_message = {
                            "role": "user",
                            "content": new_image_content
                        }
                        msg_tokens, msg_img_tokens = self._count_tokens_for_single_message(
                            temp_message, sampled_pil_images
                        )
                        new_tokens += msg_tokens
                        new_image_tokens += msg_img_tokens
            
            # Update running token counts
            current_token_count += new_tokens
            current_image_token_count += new_image_tokens

            # print("YIPPIE: ")
            # print("New turn: ", turn)
            # print("New messages: ", current_messages)
            # print(f"len(current_image_references): {len(current_image_references)}")
            # print(f"Current token count: {current_token_count} (added {new_tokens})")
            # print(f"Current image token count: {current_image_token_count} (added {new_image_tokens})")

            # add split sample when context_size is reached
            if current_messages and current_token_count > self.context_size_limit:

                # remove the latest turn and update token counts
                # We need to subtract the tokens we just added for this turn
                current_token_count -= new_tokens
                current_image_token_count -= new_image_tokens
                
                if turn["role"] in ["system", "assistant"]:
                    current_messages.pop()

                elif turn["role"] == "user":
                    if len(current_messages[-1]["content"]) == 1:  # added a new turn
                        current_messages.pop()

                    else:
                        current_messages[-1]["content"].pop()

                elif turn["role"] == "frames":
                    if len(current_messages[-1]["content"]) == sampled_frame_count:  # added a new turn
                        current_messages.pop()

                    else:
                        for _ in range(sampled_frame_count):
                            current_messages[-1]["content"].pop()

                    for _ in range(sampled_frame_count):
                        current_image_references.pop()

                # leave this turn for the next small sample
                i -= 1

                # add a system role summary prompt followed by an assistant progress summary turn
                system_summary_msg = {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Please summarize the progress."}
                    ],
                }
                current_messages.append(system_summary_msg)
                
                assistant_summary_msg = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_progress_summary}],
                }
                current_messages.append(assistant_summary_msg)
                
                # Count tokens for summary messages
                sys_tokens, sys_img_tokens = self._count_tokens_for_single_message(system_summary_msg)
                asst_tokens, asst_img_tokens = self._count_tokens_for_single_message(assistant_summary_msg)
                current_token_count += sys_tokens + asst_tokens
                current_image_token_count += sys_img_tokens + asst_img_tokens

                # Create new sample
                split_sample = {
                    "messages": current_messages,
                    "image_references": current_image_references,  # Store references instead of images
                    "sample_metadata": {
                        "sample_idx": sample.get("sample_idx", -1),
                        "video_uid": sample.get("video_uid", "unknown"),
                        "num_frames": len(current_image_references),
                    },
                }

                self.logger.info(f"Added new split sample with {current_token_count} tokens and {len(current_image_references)} images")

                self.split_samples.append(split_sample)

                # Reset for next sample
                current_messages = []
                current_image_references = []
                current_token_count = 0
                current_image_token_count = 0

                # Start new sample with updated system message
                system_msg = self._create_system_message(
                    assistant_instruction,
                    last_progress_summary,
                    sample["metadata"]["knowledge"],
                )
                new_system_message = {"role": "system", "content": system_msg}
                current_messages.append(new_system_message)
                
                # Count tokens for the new system message
                sys_tokens, sys_img_tokens = self._count_tokens_for_single_message(new_system_message)
                current_token_count += sys_tokens
                current_image_token_count += sys_img_tokens

            i += 1

        # Add unfull/remaining messages as a final sample
        if current_messages:
            split_sample = {
                "messages": current_messages,
                "image_references": current_image_references,  # Store references instead of images
                "sample_metadata": {
                    "sample_idx": sample.get("sample_idx", -1),
                    "video_uid": sample.get("video_uid", "unknown"),
                    "num_frames": len(current_image_references),
                },
            }

            self.logger.info(f"Added new unfull split sample with {current_token_count} tokens and {len(current_image_references)} images")
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