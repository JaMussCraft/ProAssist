import os
import json
import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mmassist.data.dataset import BaseDataset
from mmassist.eval.runners import BaseInferenceRunner
from mmassist.eval.eval_utils import save_json
from mmassist.eval.metrics.nlg_scorer import NLGEval


class BaseEvaluator(ABC):

    def __init__(
        self,
        model_path: str,
        dataset: BaseDataset,
        model: torch.nn.Module | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        inference_runner_type: str | None = None,
        inference_runner: BaseInferenceRunner | None = None,
        device: str | None = None,
        nlg_scorer: NLGEval = NLGEval(),
        nlg_metrics: tuple[str] = ("Bleu", "CIDEr", "METEOR"),
        not_talk_threshold: float = 0.5,
        eval_max_seq_len_str: str = "4k",
        fps: int = 2,
        force_rerun: bool = False,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.dataset = dataset
        self.dataset_name = dataset.dataset_name
        self.nlg_scorer = nlg_scorer
        self.nlg_metrics = list(nlg_metrics)
        self.not_talk_threshold = not_talk_threshold
        self.eval_max_seq_len_str = eval_max_seq_len_str
        self.eval_max_seq_len = int(eval_max_seq_len_str[:-1]) * 1024
        self.fps = fps
        self.force_rerun = force_rerun
        self.model = model
        self.tokenizer = tokenizer
        self.inference_runner_type = inference_runner_type
        self.inference_runner = inference_runner
        self.device = device
        self.set_save_dirs()

    @classmethod
    def build(cls, **kwargs) -> "BaseEvaluator":
        return cls(**kwargs)

    def update_eval_setup(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                old_v = getattr(self, k)
                if old_v != v:
                    print(f"Updating eval setup: {k}: {old_v} -> {v}")
                    setattr(self, k, v)
            else:
                raise ValueError(f"Invalid config: {k}")
        if self.inference_runner is not None:
            self.inference_runner.update_inference_setup(**kwargs)
        self.set_save_dirs()

    @property
    @abstractmethod
    def eval_name(self) -> str:
        """A unique name for the evaluation. Should be able to distinguish
        different evaluations, such as different datasets and settings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metric_names(self) -> list[str]:
        """Return the list of metric names for the evaluator."""
        raise NotImplementedError

    def set_save_dirs(self):
        """Set the save directories for the inference results and metrics.

        Folder stucture:
        model_path
        |-- eval
            |-- {eval_name1}
            |   |-- results
            |   |   |-- {sample_id}.json
            |   |   |-- ...
            |   |-- args.json
            |   |-- all_results.json
            |   |-- metrics.json
            |   |-- ...
            |-- {eval_name2}
            |   |-- ...
        Metric summarization across models storaged in a shared directory:
        {dataset}_{metric}.txt
        """
        self.eval_dir = os.path.join(self.model_path, "eval", self.eval_name)
        self.result_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.result_dir, exist_ok=True)

    def save_args(self, args_dict: dict):
        save_path = os.path.join(self.eval_dir, "args.json")
        with open(save_path, "w") as f:
            json.dump(args_dict, f, indent=4)

    @abstractmethod
    def run_prediction(self, sample_idx: int, **kwargs) -> dict:
        """Run prediction on a single video sample."""
        raise NotImplementedError

    def run_all_predictions(
        self, sample_indices: list[int], progress_bar: bool = True, **kwargs
    ) -> dict:
        """Run predictions on a list of video sample indices."""
        all_predictions = {}
        if progress_bar:
            sample_indices = tqdm(sample_indices, desc="Run predictions")
        for idx in sample_indices:
            prediction = self.run_prediction(idx, **kwargs)
            all_predictions[idx] = prediction

        return all_predictions

    def load_all_predictions(self, number_check: bool = True) -> dict:
        all_predictions = {}
        for pred_file in os.listdir(self.result_dir):
            if pred_file.endswith(".json"):
                with open(os.path.join(self.result_dir, pred_file)) as f:
                    preds = json.load(f)
                    sample_idx = int(pred_file.split(".")[0])
                    all_predictions[sample_idx] = preds

        if number_check:
            assert len(all_predictions) == len(self.dataset), (
                f"Find {len(all_predictions)} predictions in {self.result_dir}, "
                f"but the dataset has {len(self.dataset)} samples."
            )

        return all_predictions

    @abstractmethod
    def compute_metrics(self, must_complete: bool = True, **kwargs) -> dict:
        """Compute metrics for all the predictions.

        :param must_complete: If True, check if all the samples have predictions.
        """
        raise NotImplementedError
