from mmassist.eval.runners.base_runner import BaseInferenceRunner
from mmassist.eval.runners.offline_inference import OfflineInferenceRunner
from mmassist.eval.runners.stream_inference import StreamInferenceRunner, FrameOutput

runner_name_to_cls = {
    "offline": OfflineInferenceRunner,
    "stream": StreamInferenceRunner,
}

__all__ = [
    "BaseInferenceRunner",
    "OfflineInferenceRunner",
    "StreamInferenceRunner",
    "runner_name_to_cls",
    "FrameOutput",
]
