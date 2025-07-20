import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sentence_transformers as sbert

from mmassist.model import build_from_checkpoint
from mmassist.eval.runners import StreamInferenceRunner, FrameOutput
from mmassist.eval.evaluators.base_evaluator import BaseEvaluator
from mmassist.eval.evaluators.pred_match import find_match
from mmassist.eval.eval_utils import get_file_path, save_json, get_match_time_window

from mmassist.datasets.generate.llm_utils import LLMGenerator
from mmassist.eval.llm_eval import (
    DIALOG_EVALUATION_PROMPT_TEMPLATE,
    EVALUATION_SYS_PROMPT,
    LLM_EVAL_METRICS,
    parse_scores,
)


class StreamEvaluator(BaseEvaluator):

    def __init__(
        self,
        *,
        context_handling_method: str = "summarize_and_drop",
        use_gt_context: bool = False,
        sts_model_type: str = "sentence-transformers/all-mpnet-base-v2",
        match_window_time: tuple[float, float] | str = "auto",
        match_dist_func_factor: float = 0.3,
        match_dist_func_power: float = 1.5,
        sts_eval_batch_size: int = 128,
        match_semantic_score_threshold: float = 0.5,
        **kwargs,
    ):
        self.context_handling_method = context_handling_method
        self.use_gt_context = use_gt_context
        self.sts_model_type = sts_model_type
        self.sts_model = None
        self.sts_eval_batch_size = sts_eval_batch_size
        self.match_window_time = match_window_time
        self.match_dist_func_factor = match_dist_func_factor
        self.match_dist_func_power = match_dist_func_power
        self.match_semantic_score_threshold = match_semantic_score_threshold
        super().__init__(**kwargs)

    @classmethod
    def build(cls, **kwargs) -> "StreamEvaluator":
        return cls(**kwargs)

    @property
    def eval_name(self) -> str:
        dataset_name = self.dataset_name.replace("/", "-")
        name = f"{dataset_name}/stream/"
        name += f"notalk{self.not_talk_threshold}"
        name += f"-maxlen_{self.eval_max_seq_len_str}"
        if self.context_handling_method != "summarize_and_drop":
            name += f"-{self.context_handling_method}"

        # sts_name = self.sts_model_type.split("/")[-1] if self.sts_model_type else "none"
        # name += f"_fps{self.fps}" +
        # name += f"_sts-{sts_name}" +
        # name += f"_match-ws{self.match_window_size}-" +
        # name += f"-df{self.match_dist_func_factor}-" +
        # name += f"-dp{self.match_dist_func_power}" +
        return name

    def _build_sts_model(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sts_model = sbert.SentenceTransformer(
            self.sts_model_type, device=self.device
        )

    def _build_inference_runner(self) -> StreamInferenceRunner:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = build_from_checkpoint(self.model_path)
        return StreamInferenceRunner.build(
            eval_name=self.eval_name,
            model=self.model,
            tokenizer=self.tokenizer,
            fps=self.fps,
            not_talk_threshold=self.not_talk_threshold,
            eval_max_seq_len=self.eval_max_seq_len,
        )

    @property
    def metric_names(self) -> list[str]:
        metric_names = [
            "missing_rate",
            "redundant_rate",
            "mean_match_cost",
            "mean_error_rate",
        ]
        for n in self.nlg_metrics:
            if n == "Bleu":
                for i in range(1, 5):
                    metric_names.append(f"{n}_{i}")
            else:
                metric_names.append(n)
        return metric_names

    def run_prediction(self, sample_idx: int, **kwargs) -> dict:
        pred_file = get_file_path(self.result_dir, sample_idx)

        prediction = None
        if os.path.exists(pred_file) and not self.force_rerun:
            try:
                prediction = StreamInferenceRunner.load_predictions(pred_file)
            except:
                pass  # sometimes the predictions file is corrupted

        if prediction is None:
            if self.inference_runner is None:
                # lazy build the inference runner
                self.inference_runner = self._build_inference_runner()
            prediction = self.inference_runner.run_inference_on_video(
                self.dataset[sample_idx],
                use_gt_context=self.use_gt_context,
                not_talk_threshold=self.not_talk_threshold,
                output_dir=self.result_dir,
                **kwargs,
            )

        # find the match between predictions and ground truth
        if "match_result" not in prediction:
            if self.sts_model_type and self.sts_model is None:
                self._build_sts_model()

            if self.match_window_time == "auto":
                match_window_time = get_match_time_window(self.dataset_name)
            else:
                assert isinstance(self.match_window_time, tuple)
            match_window = tuple(int(t * self.fps) for t in match_window_time)
            match_result = find_match(
                prediction["predictions"],
                sts_model=self.sts_model,
                match_window=match_window,
                dist_func_factor=self.match_dist_func_factor,
                dist_func_power=self.match_dist_func_power,
                batch_size=self.sts_eval_batch_size,
            )
            prediction["match_result"] = match_result.to_json()

        # save the metrics
        prediction["predictions"] = [o.to_dict() for o in prediction["predictions"]]
        save_json(prediction, get_file_path(self.result_dir, sample_idx))

        return prediction

    def compute_metrics(
        self, must_complete: bool = True, rerun_match: bool = False, **kwargs
    ) -> dict:
        all_predictions = self.load_all_predictions(number_check=must_complete)
        if rerun_match:
            from tqdm import tqdm

            for sample_idx, prediction in tqdm(all_predictions.items()):
                if self.sts_model_type and self.sts_model is None:
                    self._build_sts_model()
                if self.match_window_time == "auto":
                    match_window_time = get_match_time_window(self.dataset_name)
                else:
                    assert isinstance(self.match_window_time, tuple)
                match_window = tuple(int(t * self.fps) for t in match_window_time)
                match_result = find_match(
                    [FrameOutput(**p) for p in prediction["predictions"]],
                    sts_model=self.sts_model,
                    match_window=match_window,
                    dist_func_factor=self.match_dist_func_factor,
                    dist_func_power=self.match_dist_func_power,
                    batch_size=self.sts_eval_batch_size,
                )
                prediction["match_result"] = match_result.to_json()
                prediction["version"] = "22"
                save_json(prediction, get_file_path(self.result_dir, sample_idx))

        results = {}
        for m in ["missed", "redundant", "matched", "match_costs", "semantic_scores"]:
            results[m] = []
        results["gen_ref"] = {}
        results["time_diff"] = []
        hyps, refs = {}, {}

        # gather the predictions
        for sample_idx, pred in all_predictions.items():
            # add ofline eval scores
            match_result = pred["match_result"]
            for m in [
                "missed",
                "redundant",
                "matched",
                "match_costs",
                "semantic_scores",
            ]:
                results[m].extend(match_result[m])

            for idx, ((g, r), s) in enumerate(
                zip(match_result["matched"], match_result["semantic_scores"])
            ):
                if s > self.match_semantic_score_threshold:
                    gidx = g["frame_idx_in_stream"]
                    ridx = r["frame_idx_in_stream"]
                    uid = f"{sample_idx}_{idx}_{gidx}<>{ridx}"
                    results["gen_ref"][uid] = [g["gen"], r["ref"], s]
                    hyps[uid] = g["gen"]
                    refs[uid] = [r["ref"]]
                    time_diff = abs(gidx - ridx) / self.fps
                    results["time_diff"].append(time_diff)

        # compute metrics
        metrics = {}

        # num_matched = len(results["matched"])
        num_matched_before_filter = len(results["matched"])
        num_matched = len(results["gen_ref"])
        num_mismatched = num_matched_before_filter - num_matched
        num_missed = len(results["missed"])
        num_redundant = len(results["redundant"])
        num_total = num_matched_before_filter + num_missed + num_redundant

        J = num_matched / num_total
        metrics["jaccard_index"] = J

        metrics["missing_rate"] = num_missed / (num_matched_before_filter + num_missed)
        metrics["redundant_rate"] = num_redundant / (
            num_matched_before_filter + num_redundant
        )

        matched_semscores = [
            s
            for s in results["semantic_scores"]
            if s >= self.match_semantic_score_threshold
        ]
        mean_sem_score = np.mean(matched_semscores) if matched_semscores else 0
        metrics["semantic_score"] = mean_sem_score
        mean_tdiff = np.mean(results["time_diff"]) if results["time_diff"] else 0
        metrics["time_diff"] = mean_tdiff

        p = num_matched / (num_matched_before_filter + num_redundant)
        r = num_matched / (num_matched_before_filter + num_missed)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        metrics["precision"] = p
        metrics["recall"] = r
        metrics["F1"] = f1

        metrics["num_matched"] = num_matched
        metrics["num_mismatched"] = num_mismatched
        metrics["num_missed"] = num_missed
        metrics["num_redundant"] = num_redundant

        # compute NLG scores
        if hyps:
            scores = self.nlg_scorer.compute_metrics(refs, hyps, self.nlg_metrics)
            for s_name, s in scores.items():
                metrics[s_name] = s
                metrics[s_name + "_w"] = J * s
        else:
            for m in self.nlg_metrics:
                if m == "Bleu":
                    for i in range(1, 5):
                        metrics[f"{m}_{i}"] = 0
                        metrics[f"{m}_{i}_w"] = 0
                else:
                    metrics[m] = 0
                    metrics[m + "_w"] = 0

        # #### DEPRECATED ###
        # # WER
        # error_rates = []
        # error_rates_v2 = []
        # precisions = []
        # recalls = []
        # f1s = []
        # semantic_scores = np.array(results["semantic_scores"])
        # # for t in np.linspace(0.05, 1.0, 20):
        # for t in np.linspace(0.5, 1.0, 11):
        #     correct = semantic_scores[semantic_scores >= t]
        #     incorrect = semantic_scores[semantic_scores < t]
        #     num_correct = len(correct)
        #     num_incorrect = len(incorrect)
        #     assert num_correct + num_incorrect == num_matched
        #     num_errors = num_incorrect + num_missed + num_redundant
        #     num_total_er = num_matched + num_missed
        #     error_rate = num_errors / num_total_er
        #     error_rates.append(error_rate)
        #     error_rate_v2 = num_errors / (num_total_er + num_redundant)
        #     error_rates_v2.append(error_rate_v2)

        #     p = num_correct / (num_matched + num_redundant)
        #     r = num_correct / (num_matched + num_missed)
        #     f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        #     precisions.append(p)
        #     recalls.append(r)
        #     f1s.append(f1)

        #     # metrics[f"ER@{t:.2f}"] = error_rate
        #     print(
        #         f"ER@{t:.2f}: {error_rate:.3f} (S={num_incorrect}, C={num_correct}, M={num_missed}, R={num_redundant})"
        #     )
        # metrics["mean_error_rate"] = np.mean(error_rates)
        # er_v2 = np.mean(error_rates_v2)
        # metrics["mean_error_rate_v2"] = er_v2
        # metrics["jaccard_index_v2"] = 1 - er_v2
        # metrics["AP"] = np.mean(precisions)
        # metrics["AR"] = np.mean(recalls)
        # metrics["Avg-F1"] = np.mean(f1s)

        # save the gathered preictions and metrics
        save_json(results, os.path.join(self.eval_dir, "all_results.json"))
        save_json(metrics, os.path.join(self.eval_dir, "metrics.json"))

        return metrics

    def llm_eval(
        self,
        llm: LLMGenerator,
        must_complete: bool = True,
        num_repeat: int = 3,
        batch_size: int = 8,
        **kwargs,
    ) -> None:
        metric_file = os.path.join(self.eval_dir, "llm_eval_metrics.json")
        if not self.force_rerun and os.path.exists(metric_file):
            print("LLM evaluation already completed. Skipping.")
            return

        all_predictions = self.load_all_predictions(number_check=must_complete)
        print(f"Running LLM evaluation on {len(all_predictions)} samples")

        results = []
        llm_eval_inputs = []
        for sample_idx, prediction in all_predictions.items():
            ref_dialog = ""
            for turn in prediction["predictions"]:
                time = turn["timestamp_in_stream"]
                for t in turn["text_inputs"]:
                    if t[0] == "user":
                        ref_dialog += f"[{time}s] User: {t[1]}\n"
                if turn["ref"]:
                    ref_dialog += f"[{time}s] Assistant: {turn['ref']}\n"

            gen_dialog = ""
            for turn in prediction["predictions"]:
                time = turn["timestamp_in_stream"]
                for t in turn["text_inputs"]:
                    if t[0] == "user":
                        gen_dialog += f"[{time}s] User: {t[1]}\n"
                if turn["gen"]:
                    gen_dialog += f"[{time}s] Assistant: {turn['gen']}\n"

            prompt = DIALOG_EVALUATION_PROMPT_TEMPLATE.format(
                reference_dialog=ref_dialog, generated_dialog=gen_dialog
            )
            inputs = [("system", EVALUATION_SYS_PROMPT), ("user", prompt)]
            llm_eval_inputs.append(inputs)
            results.append(
                {"sample_idx": sample_idx, "ref": ref_dialog, "gen": gen_dialog}
            )

        save_dir = os.path.join(self.eval_dir, "llm_eval_outputs")
        os.makedirs(save_dir, exist_ok=True)

        # LLM generation
        for idx in tqdm(range(0, len(llm_eval_inputs), batch_size)):
            batch_prompts = llm_eval_inputs[idx : idx + batch_size]
            batch_results = results[idx : idx + batch_size]
            batch_outputs = llm.batch_generate(batch_prompts, n=num_repeat)

            for result, outputs in zip(batch_results, batch_outputs):
                result["llm_outputs"] = outputs
                sample_idx = result["sample_idx"]

                all_scores = {k: [] for k in LLM_EVAL_METRICS}
                for n in range(num_repeat):
                    parsed_dict = parse_scores(outputs[n])
                    if parsed_dict:
                        for k, v in parsed_dict.items():
                            all_scores[k].append(v)

                mean_scores = {}
                for k, v in all_scores.items():
                    if v:
                        mean_scores[k] = sum(v) / len(v)
                    else:
                        print(f"Warning: no valid score for sample {sample_idx} - {k}")
                        mean_scores[k] = None

                result["all_scores"] = all_scores
                result["scores"] = mean_scores

                save_json(result, get_file_path(save_dir, sample_idx))

        # gather the metrics
        metrics = {}
        for m in LLM_EVAL_METRICS:
            scores = [r["scores"][m] for r in results if r["scores"][m] is not None]
            metrics[m] = np.mean(scores) if scores else 0.0

        # save the results and metrics
        save_json(results, os.path.join(self.eval_dir, "llm_eval_results.json"))
        save_json(metrics, metric_file)
