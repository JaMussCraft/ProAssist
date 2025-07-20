import os
import json
import torch
import numpy as np

from mmassist.model import build_from_checkpoint
from mmassist.eval.runners import OfflineInferenceRunner
from mmassist.eval.evaluators.base_evaluator import BaseEvaluator
from mmassist.eval.eval_utils import get_file_path, save_json


class OfflineEvaluator(BaseEvaluator):

    @classmethod
    def build(cls, **kwargs) -> "OfflineEvaluator":
        return cls(**kwargs)

    @property
    def eval_name(self) -> str:
        dataset_name = self.dataset_name.replace("/", "-")
        name = f"{dataset_name}/offline/"
        name += f"notalk{self.not_talk_threshold}"
        name += f"-maxlen_{self.eval_max_seq_len_str}"
        return name

    def _build_inference_runner(self) -> OfflineInferenceRunner:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = build_from_checkpoint(self.model_path)
        return OfflineInferenceRunner.build(
            eval_name=self.eval_name,
            model=self.model,
            tokenizer=self.tokenizer,
            not_talk_threshold=self.not_talk_threshold,
            eval_max_seq_len=self.eval_max_seq_len,
        )

    @property
    def metric_names(self) -> list[str]:
        metric_names = ["lm_ppl", "time_diff", "fluency", "token_acc"]
        for p in ["dialog", "summary"]:
            for n in self.nlg_metrics:
                if n == "Bleu":
                    for i in range(1, 5):
                        metric_names.append(f"{p}_{n}_{i}")
                else:
                    metric_names.append(f"{p}_{n}")
        return metric_names

    def run_prediction(self, sample_idx: int, **kwargs) -> dict:
        pred_file = get_file_path(self.result_dir, sample_idx)

        prediction = None
        if os.path.exists(pred_file) and not self.force_rerun:
            try:
                prediction = OfflineInferenceRunner.load_predictions(pred_file)
            except:
                # sometimes the predictions file is corrupted
                pass

        if prediction is None:
            if self.inference_runner is None:
                # lazy build the inference runner
                self.inference_runner = self._build_inference_runner()
            prediction = self.inference_runner.run_inference_on_video(
                self.dataset[sample_idx],
                eval_max_seq_len=self.eval_max_seq_len,
                not_talk_threshold=self.not_talk_threshold,
                output_dir=self.result_dir,
                **kwargs,
            )

        return prediction

    def compute_metrics(self, must_complete: bool = True, **kwargs) -> dict:
        all_predictions = self.load_all_predictions(number_check=must_complete)

        results = {}
        for m in ["lm_ppl", "time_diff", "fluency", "token_acc"]:
            results[m] = []
        results["gen_ref"] = {}
        hyps = {}
        refs = {}

        # gather the predictions
        num_overflowed = 0
        for sample_idx, pred in all_predictions.items():
            if pred["overflow"]:
                num_overflowed += 1
            pred["time_diff"] = pred["frame_diff"] / self.fps

            # add ofline eval scores
            for m in ["lm_ppl", "time_diff", "fluency", "token_acc"]:
                results[m].append(pred[m])

            # collect the generated text
            for turn_idx, turn in enumerate(pred["conversation"]):
                if "gen" in turn:
                    uid = f"{sample_idx}_{turn_idx}"
                    hyps[uid] = turn["gen"]
                    refs[uid] = [turn["content"]]
                    results["gen_ref"][uid] = [turn["gen"], turn["content"]]

        num_total = len(all_predictions)
        ratio = num_overflowed / num_total
        print(f"Overflowed ratio: {num_overflowed} / {num_total} ({ratio:.1%})")

        # compute metrics
        metrics = {}
        for m in ["lm_ppl", "time_diff", "fluency", "token_acc"]:
            metrics[m] = np.mean(results[m])
        if hyps:
            scores = self.nlg_scorer.compute_metrics(refs, hyps, self.nlg_metrics)
            for s_name, s in scores.items():
                metrics[s_name] = s
        metrics["overflow_ratio"] = ratio

        # save the gathered preictions and metrics
        save_json(results, os.path.join(self.eval_dir, "all_results.json"))
        save_json(metrics, os.path.join(self.eval_dir, "metrics.json"))

        return metrics
