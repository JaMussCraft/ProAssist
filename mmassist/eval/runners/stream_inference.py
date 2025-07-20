import os
import json
import torch
from dataclasses import dataclass, asdict
from PIL import Image
from transformers import PreTrainedTokenizer

from mmassist.model.tokenization_proact import MultimodalChat
from mmassist.model.modeling_proact import (
    ProActModelMixin,
    KV_CACHE,
    trim_past_key_values,
    ExceedContextHandling,
)
from mmassist.data.utils import tensor_to_pil_images
from mmassist.eval.runners.base_runner import BaseInferenceRunner
from mmassist.eval.demo_utils import annotate_and_save_video
from mmassist.eval.eval_utils import get_file_path, save_json
from mmassist.datasets.prepare.prompts import summarize_query


@dataclass
class FramesInput:
    """Input unit to be consumed by the StreamInferenceRunner for
    a single-step inference. Should be obtained from a video stream
    using the StreamProcessor.
    """

    input_str: str
    """The formatted input text string for the current frame(s) including
    the image placeholder tokens and formatted user/system messages (optional).
    """
    model_inputs: dict
    """Inputs to the model. Should include the input_ids and the image tensors
    in either the 'images' or 'image_embeds' key.
    """
    num_frames: int = 1
    """Number of frames in the input. Should always be 1 when evaluate with
    pre-recorded videos, but can be more than 1 in live demo scenarios."""
    images: list[Image.Image] | None = None
    """Images for creating annotated videos."""
    frame_idxs_in_stream: list[int] | None = None
    """The index of the frame in the input stream."""
    frame_idxs_in_original_video: list[int] | None = None
    """The index of the frame in the original video."""
    timestamps: list[float] | None = None
    """The timestamp of the last frame in seconds."""
    ref_output_str: str | None = None
    """The reference model output for this input."""
    input_messsages: list[dict[str, str]] | None = None
    """The input messages for the current frame(s)."""


@dataclass
class FrameOutput:
    """Per-frame output from the StreamInferenceRunner.
    Can be used to create annotated videos or compute metrics.
    """

    gen: str
    """The generated text output."""
    ref: str | None = None
    """The reference text output."""
    image: Image.Image | None = None
    """The frame image. Useful for creating annotated videos."""
    text_inputs: list[tuple[str, str]] | None = None
    """The input texts for the current frame in the format (role, text). 
    Role can be either 'user' or 'system'."""
    frame_idx_in_stream: int | None = None
    """The index of the frame in the video."""
    frame_idx_in_original_video: int | None = None
    """The index of the frame in the original video."""
    timestamp_in_stream: float | None = None
    """The timestamp of the frame in seconds."""

    def to_dict(self, ignore_keys: str | list[str] = "image") -> dict:
        ret = asdict(self)
        if ignore_keys:
            if isinstance(ignore_keys, str):
                ignore_keys = [ignore_keys]
            for k in ignore_keys:
                ret.pop(k, None)
        return ret


@dataclass
class StreamProcessor:
    tokenizer: PreTrainedTokenizer
    chat_formatter: MultimodalChat
    fps: int = 2

    def get_input_sequence(
        self, num_images: int, messages: list[dict[str, str]], first_turn: bool = False
    ) -> tuple[torch.LongTensor, str]:
        if messages:
            if first_turn:
                input_str_txt = self.chat_formatter.apply_chat_template(messages)
            else:
                input_str_txt = ""
                for m in messages:
                    input_str_txt += self.chat_formatter.add_message(m)
        else:
            input_str_txt = ""
        input_str_img = self.chat_formatter.add_img_tokens(num_images)
        input_str = input_str_txt + input_str_img
        input_ids = self.tokenizer(
            input_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return input_ids, input_str

    def add_last_assistant_message(
        self,
        model_inputs: dict[str, torch.Tensor],
        last_msg: str | torch.LongTensor | None,
    ) -> dict[str, torch.Tensor]:
        if last_msg is None:
            return model_inputs
        input_ids = model_inputs["input_ids"]
        if isinstance(last_msg, str):
            last_msg = self.tokenizer(
                last_msg, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
        input_ids = torch.cat([last_msg.to(input_ids), input_ids], dim=-1)
        model_inputs["input_ids"] = input_ids
        return model_inputs

    def processed_conv_data_to_stream(self, video: dict) -> list[FramesInput]:
        conversation = video["conversation"]
        frame_index_offset = video["start_frame_idx"]
        images = video.get("images", None)
        encoded_images = video.get("encoded_images", None)

        frame_streams = []
        last_msgs = []
        for turn_idx, turn in enumerate(conversation):
            if turn["role"] in ["system", "user"]:
                last_msgs.append(turn)
            elif turn["role"] == "assistant":
                pass
                # last_msgs = []
            else:
                start = turn["start"] - frame_index_offset
                end = turn["end"] - frame_index_offset

                for k in range(start, end):
                    # prepare appended input for the model
                    num_images = 1
                    input_ids, input_str = self.get_input_sequence(
                        num_images, last_msgs, not frame_streams
                    )
                    model_inputs = {"input_ids": input_ids}
                    pil_img = []
                    if images is not None:
                        pt_img = images[k : k + 1]
                        model_inputs["images"] = pt_img
                        pil_img = tensor_to_pil_images(pt_img)
                    if encoded_images is not None:
                        model_inputs.pop("images", None)
                        image_embeds = encoded_images[k : k + 1].flatten(0, 1)
                        model_inputs["image_embeds"] = image_embeds

                    # get the reference prediction for this frame
                    ref_output_str = self.tokenizer.img_sep_token
                    if k == end - 1 and turn_idx < len(conversation) - 1:
                        next_turn = conversation[turn_idx + 1]
                        if next_turn["role"] == "assistant":
                            ref_output_str = self.chat_formatter.add_message(next_turn)

                    time = (len(frame_streams) + 1) / self.fps
                    frame = FramesInput(
                        input_str=input_str,
                        model_inputs=model_inputs,
                        images=pil_img,
                        num_frames=num_images,
                        ref_output_str=ref_output_str,
                        frame_idxs_in_stream=[len(frame_streams)],
                        frame_idxs_in_original_video=[k + frame_index_offset],
                        timestamps=[time],
                        input_messsages=last_msgs,
                    )
                    frame_streams.append(frame)
                    last_msgs = []

        return frame_streams

    def cleanup_text(self, text: str) -> tuple[str, str | None]:
        # print(text)
        if self.chat_formatter.eot in text:
            # remove the part after EOS, if any
            text = text.split(self.chat_formatter.eot, 1)[0]
        # remove the bot token
        text = text.strip().replace(self.chat_formatter.bot, "")
        # remove the image tokens
        text = text.replace(self.chat_formatter.img_token, "").replace(
            self.chat_formatter.sep_token, ""
        )
        if self.chat_formatter.eor in text:
            # extract the role and the text
            role, text = text.split(self.chat_formatter.eor, 1)
            role = role.replace(self.chat_formatter.bor, "")
            return text, role
        if text == ".":
            text = ""
        return text, None


class StreamInferenceRunner(BaseInferenceRunner):

    def __init__(self, fps: int = 2, reserved_seq_len: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.fps = fps
        self.processor = StreamProcessor(self.tokenizer, self.chat_formatter, fps)
        self.reserved_seq_len = reserved_seq_len
        self.ctxlen_to_reduce_kv_cache = self.eval_max_seq_len - reserved_seq_len
        self.last_keep_len = 512  # used in DROP_MIDDLE strategy
        self.initial_sys_prompt = None
        self.knowledge = None
        self.init_key_values_to_keep = None

    @classmethod
    def build(cls, fps: int, **kwargs) -> "StreamInferenceRunner":
        return super().build(fps=fps, **kwargs)

    def run_inference_on_video(
        self,
        video: dict,
        streams: list[FramesInput] | None = None,
        use_gt_context: bool = False,
        max_time: int = -1,
        output_dir: str = "",
        video_output_dir: str = "",
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """Run inference on a video conversation.

        :param video: The video conversation data.
        :param streams: The preprocessed stream data. If not provided, the method
            will preprocess the conversation data.
        :param use_gt_context: Whether to use the ground truth context. If True,
            the model will use the ground truth assistant responses in the data
            instead of its own generated responses as context.
        :param max_time: The maximum time in the stream. The inference will stop
            after this time is reached, even if not reach the end of the video.
        :param output_dir: The directory to save the predictions and metadata.
            Default: Do not save the outputs.
        :param video_output_dir: The directory to save an annotated video. Must
            provide the images in the video data. Default: Do not save the video.
        :param verbose: Whether to print the outputs for debugging.
        :param kwargs: Additional arguments to pass to the model.

        :return: a dict of the predictions and metadata.
        """
        if streams is None:
            if verbose:
                print("Preprocessing stream data...")
            streams = self.processor.processed_conv_data_to_stream(video)
        # print(streams)
        past_key_values = None
        last_msg = None
        outputs = []
        kwargs["drop_generated_kv_cache"] = use_gt_context
        if kwargs.get("not_talk_threshold") is None:
            kwargs["not_talk_threshold"] = self.not_talk_threshold
        try:
            assert streams[0].input_messsages[0]["role"] == "system"
            self.initial_sys_prompt = streams[0].input_messsages[0]["content"]
        except:
            self.initial_sys_prompt = ""
        self.knowledge = ""
        self.init_key_values_to_keep = None
        is_first_user_input = False
        for frames in streams:

            # print("input_ids before add last turn:", frames.model_inputs["input_ids"])
            if not self.knowledge:
                for msg in frames.input_messsages:
                    if msg["role"] == "system" and "Task knowledge:" in msg["content"]:
                        self.knowledge = msg["content"]
            if self.init_key_values_to_keep is None:
                for msg in frames.input_messsages:
                    if msg["role"] == "user":
                        is_first_user_input = True

            # prepare the input for the current frame(s)
            model_inputs = self.processor.add_last_assistant_message(
                frames.model_inputs, last_msg
            )

            for k, v in model_inputs.items():
                if isinstance(v, torch.FloatTensor):
                    model_inputs[k] = v.to(self.model.device, self.model.dtype)
                else:
                    model_inputs[k] = v.to(self.model.device)

            ref_out = frames.ref_output_str
            if use_gt_context and ref_out is None:
                print("Warning: ground truth not provided, set use_gt_context to False")
                use_gt_context = False

            # run the model
            input_embeds = self.model.joint_embed(**model_inputs)

            # print("input_ids after add last turn:", model_inputs["input_ids"])
            # print("len input_ids:", frames.model_inputs["input_ids"].shape[1])
            # print("len input_embeds:", len(input_embeds[0]))
            # print(
            #     "len kv cache:",
            #     past_key_values[0][0].shape[2] if past_key_values is not None else 0,
            # )

            output_ids, past_key_values = self.model.fast_greedy_generate(
                input_embeds, past_key_values, verbose=False, **kwargs
            )
            if is_first_user_input:
                self.init_key_values_to_keep = past_key_values
                is_first_user_input = False

            # prepare the output
            gen_text_raw = self.processor.tokenizer.decode(output_ids[0])
            text_inputs = [(t["role"], t["content"]) for t in frames.input_messsages]
            cleaned_gen_text = self.processor.cleanup_text(gen_text_raw)[0]
            cleaned_ref_text = None
            if ref_out is not None:
                cleaned_ref_text = self.processor.cleanup_text(ref_out)[0]

            # if there are multiple input frames, add dummy intermediate outputs to
            # ensure we have per-frame outputs; this may only happen in live demo
            for i in range(frames.num_frames):
                img = frames.images[i] if frames.images else None
                frame_idx_in_stream = (
                    frames.frame_idxs_in_stream[i]
                    if frames.frame_idxs_in_stream
                    else None
                )
                frame_idx_in_original_video = (
                    frames.frame_idxs_in_original_video[i]
                    if frames.frame_idxs_in_original_video
                    else None
                )
                timestamp = frames.timestamps[i] if frames.timestamps else None
                gen = "" if i < frames.num_frames - 1 else cleaned_gen_text
                ref = None if i < frames.num_frames - 1 else cleaned_ref_text
                text_inputs = None if i < frames.num_frames - 1 else text_inputs
                outputs.append(
                    FrameOutput(
                        gen=gen,
                        ref=ref,
                        image=img,
                        text_inputs=text_inputs,
                        frame_idx_in_stream=frame_idx_in_stream,
                        frame_idx_in_original_video=frame_idx_in_original_video,
                        timestamp_in_stream=timestamp,
                    )
                )

            # set the last message to be appended in the next input
            if use_gt_context:
                last_msg = ref_out
            else:
                # Note that the generated text is already included in the kv cache
                # so we don't need to append it to the next turn's input_ids, but
                # only the last generated token (an EOS or a separator token).
                if self.processor.tokenizer.img_sep_token_id is None:
                    last_msg = None
                else:
                    last_msg = output_ids[:, -1:]

            if verbose:
                if text_inputs or cleaned_ref_text or cleaned_gen_text:
                    for role, text in text_inputs:
                        print(f"[{timestamp:.1f}s] {role.upper()}: {text}")
                    print(f"[{timestamp:.1f}s] REF : {cleaned_ref_text}")
                    print(f"[{timestamp:.1f}s] GEN : {cleaned_gen_text}")
                    print(f"Context Length: {past_key_values[0][0].shape[2]}")

            # kv cache management
            past_key_values, last_msg = self.manage_kv_cache(
                frames, past_key_values, last_msg
            )

            if max_time > 0 and timestamp >= max_time:
                if verbose:
                    print(f"Max time reached: {max_time}s")
                break

        dataset = video["dataset"]
        sample_idx = video["sample_idx"]

        ctx_hand_strategy = self.model.config.exceed_context_handling_stragety.name

        metadata = {
            "dataset": dataset,
            "sample_idx": video["sample_idx"],
            "video_uid": video["video_uid"],
            "fps": self.processor.fps,
            "use_gt_context": use_gt_context,
            "no_talk_threshold": self.not_talk_threshold,
            "ctxlen_to_reduce_kv_cache": self.ctxlen_to_reduce_kv_cache,
            "context_handling_strategy": ctx_hand_strategy,
        }
        if output_dir:
            seraliazible = [o.to_dict() for o in outputs]
            save_dict = {"metadata": metadata, "predictions": seraliazible}
            save_json(save_dict, get_file_path(output_dir, sample_idx, "json"))

        if video_output_dir:
            if any(o.image is None for o in outputs):
                print("Cannot save video as some images are None.")
                return outputs
            outputs = [o.to_dict(ignore_keys="") for o in outputs]
            save_file = get_file_path(video_output_dir, sample_idx, "mp4")
            annotate_and_save_video(outputs, save_file, self.processor.fps)

        predictions = {"metadata": metadata, "predictions": outputs}
        return predictions

    @staticmethod
    def load_predictions(file: str) -> dict:
        """Load the predictions from json.

        :param file: The json file saved using `run_inference_on_video` method.
        :return: The list of FrameOutput objects and the metadata.
        """
        with open(file, "r") as f:
            data = json.load(f)
        data["predictions"] = [FrameOutput(**o) for o in data["predictions"]]
        return data

    def generate_progress_summary(
        self,
        model_inputs: dict[str, torch.Tensor],
        past_key_values: KV_CACHE,
        max_length: int,
    ) -> str:
        input_embeds = self.model.joint_embed(**model_inputs)
        output_ids, past_key_values = self.model.fast_greedy_generate(
            input_embeds, past_key_values, max_length=max_length
        )
        gen_text_raw = self.processor.tokenizer.decode(output_ids[0])
        cleaned_gen_text = self.processor.cleanup_text(gen_text_raw)[0]
        return cleaned_gen_text

    def manage_kv_cache(
        self,
        frames: FramesInput,
        past_key_values: KV_CACHE,
        last_msg: str | torch.LongTensor | None,
    ) -> tuple[KV_CACHE | None, str | None]:

        curr_seq_len = past_key_values[0][0].shape[2]
        if curr_seq_len < self.ctxlen_to_reduce_kv_cache:
            return past_key_values, last_msg

        # need to use some strategy to reduce the cache size
        strategy = self.model.config.exceed_context_handling_stragety
        if strategy == ExceedContextHandling.DROP_ALL:
            # print("Drop all!")
            return None, last_msg
        elif strategy == ExceedContextHandling.DROP_MIDDLE:
            init_kv_cache = self.init_key_values_to_keep
            if init_kv_cache is None:
                return None, last_msg
            init_kv_cache_len = init_kv_cache[0][0].shape[2]
            if curr_seq_len < self.last_keep_len:
                # no need to drop
                return past_key_values, last_msg
            start = curr_seq_len - self.last_keep_len
            if start < init_kv_cache_len:
                start = init_kv_cache_len
            # print(f"start: {start}, end: {curr_seq_len}")
            last_kv_cache = trim_past_key_values(past_key_values, start, curr_seq_len)
            new_kv_cache = []
            for (init_keys, init_values), (last_keys, last_values) in zip(
                init_kv_cache, last_kv_cache
            ):
                new_kv_cache.append(
                    (
                        torch.cat([init_keys, last_keys], dim=2),
                        torch.cat([init_values, last_values], dim=2),
                    )
                )
            new_kv_cache_len = new_kv_cache[0][0].shape[2]
            # print(f"Drop middle! New kv cache len: {new_kv_cache_len}")
            return tuple(new_kv_cache), last_msg
        elif strategy == ExceedContextHandling.SUMMARIZE_AND_DROP:
            model_inputs = frames.model_inputs
            num_frames = frames.num_frames
            input_ids, _ = self.processor.get_input_sequence(
                num_frames, [summarize_query], first_turn=False
            )
            model_inputs["input_ids"] = input_ids
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

            summary = self.generate_progress_summary(
                model_inputs, past_key_values, max_length=512
            )
            print("IPS envoked - Generated summary:", summary)
            if self.initial_sys_prompt:
                summary = f"{self.initial_sys_prompt} {summary}"
            if self.knowledge:
                summary = f"{summary} {self.knowledge}"
            last_msg = self.processor.chat_formatter.apply_chat_template(
                [{"role": "system", "content": summary}]
            )
            return None, last_msg

        raise ValueError(f"Unsupported exceed_context_handling: {strategy}")
