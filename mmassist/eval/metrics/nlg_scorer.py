# Modified from: https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/__init__.py
# Copyright (c) Microsoft Corporation. All rights reserved.
# Original license: MIT
from .bleu.bleu import Bleu
from .cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from .tokenizer import PTBTokenizer


class NLGEval(object):
    valid_metrics = {
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "CIDEr",
    }

    def __init__(self, metrics_to_omit: set[str] | None = None):
        """
        :param metrics_to_omit: Default: Use all metrics. See `NLGEval.valid_metrics` for all metrics.
            The previous parameters will override metrics in this one if they are set.
            Metrics to omit. Omitting Bleu_{i} will omit Bleu_{j} for j>=i.
        :type metrics_to_omit: Optional[Collection[str]]
        """

        if metrics_to_omit is None:
            self.metrics_to_omit = set()

        self.scorers = {}
        if "Bleu" not in self.metrics_to_omit:
            self.scorers["Bleu"] = (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        if "CIDEr" not in self.metrics_to_omit:
            self.scorers["CIDEr"] = (Cider(), "CIDEr")
        if "METEOR" not in self.metrics_to_omit:
            self.scorers["METEOR"] = (Meteor(), "METEOR")

        self.tokenizer = PTBTokenizer()

    def compute_metrics(
        self,
        refs: dict[str, list[str]],
        hyps: dict[str, str],
        metrics_to_eval: list | None = None,
    ) -> list[dict[str, float]]:
        """Compute the NLG metrics between a single hypothesis and a list of reference
        sentences, or between a list of hypotheses and a list of lists of reference.

        :param refs: dict of reference sentences with image ID as key and list of
            reference sentences as value
        :param hyps: dict of hypothesis sentences with image ID as key and hypothesis
            sentences as value.
        :param metrics: Optional. List of metrics to compute. Default: Use all metrics.
        :return: dict of metrics with metric names as key and scores as value.
        """

        assert len(refs) == len(hyps), (
            f"The number of hypotheses {len(hyps)} does not "
            f"match the number of references {len(refs)}"
        )

        # tokenize hyps and refs
        hyps = {idx: [hyp] for idx, hyp in hyps.items()}
        hyps = self.tokenizer.tokenize(hyps)
        refs = self.tokenizer.tokenize(refs)

        # filter out empty references
        # NOTE: this may only happen due to poor data synthesis
        valid_refs, valid_hyps = {}, {}
        for idx, hyp in hyps.items():
            ref = refs[idx]
            if any(len(r) == 0 for r in ref):
                continue
            valid_refs[idx] = ref
            valid_hyps[idx] = hyp

        ret_scores = {}
        for scorer_name, (scorer, metrics) in self.scorers.items():
            if metrics_to_eval is not None and scorer_name not in metrics_to_eval:
                continue
            if not valid_refs or not valid_hyps:
                score = [0.0] * len(metrics) if isinstance(metrics, list) else 0.0
            else:
                try:
                    score, _ = scorer.compute_score(valid_refs, valid_hyps)
                except:
                    print(refs)
                    print(hyps)
                    raise Exception(f"Error computing {scorer_name} scores")

            # save the scores
            if isinstance(metrics, list):
                for sc, metric in zip(score, metrics):
                    ret_scores[metric] = sc
            else:
                metric = metrics
                ret_scores[metric] = score

        return ret_scores
