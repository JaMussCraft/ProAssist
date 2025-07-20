import os
from abc import ABC, abstractmethod
from typing import Any
from transformers import PreTrainedTokenizer

from mmassist.model.modeling_proact import ProActModelMixin
from mmassist.model.tokenization_proact import MultimodalChat


class BaseInferenceRunner(ABC):
    def __init__(
        self,
        eval_name: str,
        model: ProActModelMixin,
        tokenizer: PreTrainedTokenizer,
        not_talk_threshold: float,
        eval_max_seq_len: int = 512,
        **kwargs,
    ) -> None:
        self.eval_name = eval_name
        self.model = model
        self.tokenizer = tokenizer
        if not hasattr(tokenizer, "chat_formatter") or not isinstance(
            tokenizer.chat_formatter, MultimodalChat
        ):
            raise ValueError(
                "Must provide a tokenizer with a `MultimodalChat` chat_formatter."
            )
        self.chat_formatter: MultimodalChat = tokenizer.chat_formatter
        self.not_talk_threshold = not_talk_threshold
        self.eval_max_seq_len = eval_max_seq_len

    @classmethod
    def build(
        cls,
        eval_name: str,
        model: ProActModelMixin,
        tokenizer: PreTrainedTokenizer,
        not_talk_threshold: float,
        eval_max_seq_len: int,
        **kwargs,
    ) -> "BaseInferenceRunner":
        return cls(
            eval_name=eval_name,
            model=model,
            tokenizer=tokenizer,
            not_talk_threshold=not_talk_threshold,
            eval_max_seq_len=eval_max_seq_len,
            **kwargs,
        )

    def update_inference_setup(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def run_inference_on_video(self, video: dict, output_dir: str, **kwargs) -> Any:
        """Run inference on a video data sample and save the results."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_predictions(file: str) -> Any:
        """Load the model from the checkpoint."""
        raise NotImplementedError
