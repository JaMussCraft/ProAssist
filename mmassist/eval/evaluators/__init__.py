from mmassist.eval.evaluators.base_evaluator import BaseEvaluator
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator
from mmassist.eval.evaluators.offline_evaluator import OfflineEvaluator

evaluator_name_to_cls: dict[str, OfflineEvaluator | StreamEvaluator] = {
    "offline": OfflineEvaluator,
    "stream": StreamEvaluator,
}
