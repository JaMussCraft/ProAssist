import os
import json
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser
from transformers import TrainingArguments as HFTrainingArguments

DATA_ROOT_DIR = os.environ.get("DATA_ROOT_DIR", "<please_set_me>")
# raw data dir: {DATA_ROOT_DIR}/datasets
# processed data dir: {DATA_ROOT_DIR}/processed_data


@dataclass
class ModelArguments:
    llm_pretrained: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "The path to load a pretrained LLM model."},
    )
    vision_pretrained: str | None = field(
        default="google/siglip-so400m-patch14-384",
        metadata={"help": "The path to load a pretrained vision encoder."},
    )
    vision_hidden_size: int = field(
        default=1152,
        metadata={"help": "Dimension of the visual feature."},
    )
    img_resolution: int | None = field(
        default=None,
        metadata={
            "help": (
                "The resolution of ipnut image. If None, will use the "
                "preprocess setup during visual model preptrainig."
            )
        },
    )
    use_img_cls_token: bool = field(
        default=True,
        metadata={"help": "Whether to use the [CLS] token from CLIP."},
    )
    img_patch_token_size: int = field(
        default=0,
        metadata={
            "help": (
                "The size of patch tokens to use from CLIP. The original patch "
                "feature map will be 2D-pooled to this size."
            )
        },
    )
    img_patch_token_layer: int = field(
        default=-2,
        metadata={"help": "The layer to extract patch tokens from CLIP."},
    )
    img_sep_token: str = field(
        default="",
        metadata={"help": "Separator between frames. '' or 'none' for not use."},
    )
    max_seq_len: int = field(
        default=4096,
        metadata={"help": "The maximum input token sequence length."},
    )
    padding_side: str = field(
        default="right",
        metadata={"help": "Where to pad.", "choices": ["left", "right"]},
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "HF argument to specify the attention implmenetation type."},
    )
    w2t_logit_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "The weight for predicting the 'not-talk' logit. Only take effect"
                " when use_binary_decision_head is FALSE."
            )
        },
    )
    use_binary_decision_head: bool = field(
        default=False,
        metadata={"help": "Whether to use a binary decision head for whether to talk."},
    )
    binary_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "The weight for the binary decision loss. Only take effect when "
                "use_binary_decision_head is TRUE."
            )
        },
    )
    binary_decision_head_type: str = field(
        default="linear",
        metadata={
            "help": "The type of the binary decision head.",
            "choices": ["linear", "mlp"],
        },
    )

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingArguments(HFTrainingArguments):
    data_root_dir: str = field(default=os.path.join(DATA_ROOT_DIR, "processed_data"))
    train_datasets: str = field(
        default="ego4d/narration_train_L2048_I1+SEP",
        metadata={
            "help": (
                "Training datasets seperated by comma. Each dataset should be in "
                r"the format of {dataset_dir}/{annotation_file}"
            )
        },
    )
    eval_datasets: str | None = field(
        default=None,
        metadata={
            "help": (
                "Evaluation datasets seperated by comma. Each dataset should be in "
                r"the format of {dataset_dir}/{annotation_file}"
            )
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "The path to load a checkpoint."},
    )
    neg_frame_sampling_rate: float = field(
        default=1.0,
        metadata={
            "help": "The rate to sample negative (no-talk) frames to compute loss."
        },
    )
    use_pose: bool = field(
        default=False,
        metadata={"help": "Whether to use PoSE for position id extension."},
    )
    llm_train_mode: str = field(
        default="lora",
        metadata={
            "help": "The training mode for LLM.",
            "choices": ["lora", "full", "frozen"],
        },
    )
    lora_modules: str = (
        "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    )
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: str = field(
        default="mm_projector",
        metadata={"help": "Modules to finetune, seperated by comma."},
    )
    output_dir: str = "outputs/debug"
    is_debug: bool = field(default=False)


@dataclass
class EvalArguments:
    model_path: str = field(
        metadata={"help": "The path to the model checkpoint to evaluate."}
    )
    inference_setups: str = field(
        metadata={
            "help": (
                "The inference configuration string. The format is "
                "<dataset>|<evaluator>|<eval_max_seq_len>|<not_talk_threshold>|"
                "<context_handling_strategy> where:"
                "<dataset> is the dataset specifier in the same format as training. "
                "<evaluator> is one of ['offline', 'stream']."
                "<eval_max_seq_len> is the maximum input token sequence length for "
                "inference, one of ['2k', '4k', '8k', '16k', '32k', '64k', '128k']. "
                "<not_talk_threshold> is the threshold of the not-to-talk probability, "
                "above which the frame is considered as no-talk. Range from 0 to 1. "
                "<context_handling_strategy> is the strategy to handle the exceed "
                "context during stream inference. One of ['drop_all', 'drop_middle' "
                ", 'summarize_and_drop']. "
                "The setups are seperated by comma, such as |||||, |||||, |||||."
            )
        },
    )
    data_root_dir: str = field(default=os.path.join(DATA_ROOT_DIR, "processed_data"))
    force_rerun: bool = field(
        default=False,
        metadata={
            "help": "Whether to force rerun the evaluation ignoring existing results."
        },
    )

    # Stream evaluation arguments
    fps: int = field(default=2, metadata={"help": "The frame rate of the video data."})
    sts_model_type: str = field(
        default="sentence-transformers/all-mpnet-base-v2",
        metadata={
            "help": (
                "The sentence transformer model to use for computing the semantic "
                "text similarity. Have to be one of the models from this list:"
                "https://www.sbert.net/docs/sentence_transformer/pretrained_models.html"
            )
        },
    )
    match_dist_func_factor: float = field(
        default=0.3,
        metadata={
            "help": (
                "The factor to control the tradeoff between the sts cost and the "
                "time difference cost in the bipartite matching."
            )
        },
    )
    match_dist_func_power: float = field(
        default=1.5,
        metadata={"help": ("The power of the distance cost function.")},
    )

    # LLM evaluation arguments
    number_repeat: int = field(
        default=3,
        metadata={"help": "The number of times to repeat LLM scoring."},
    )

    def to_dict(self):
        return asdict(self)


@dataclass
class SlurmArguments:
    job_name: str
    num_nodes: int = 1
    tasks_per_node: int = 8
    gpus_per_node: int = 8  # H100
    cpus_per_node: int = 192  # H100
    mem_gb: int = 1800  # per node
    timeout_min: int = 1440
    partition: str = "q1"
    account: str = ""
    log_dir: str = "slurm_logs/%j"
    slurm_exclude: str = ""


def parse_args(no_args: bool = False) -> tuple[ModelArguments, TrainingArguments]:
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if no_args:
        model_args, train_args, remaining = parser.parse_args_into_dataclasses(
            args=None, return_remaining_strings=True
        )
    else:
        model_args, train_args, remaining = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
    if (
        model_args.use_binary_decision_head
        and "binary_decision_head" not in train_args.finetune_modules
    ):
        train_args.finetune_modules += ",binary_decision_head"
    if model_args.img_sep_token.lower() == "none":
        model_args.img_sep_token = ""
    if train_args.is_debug:
        train_args.report_to = []
        train_args.dataloader_num_workers = 0
        train_args.dataloader_prefetch_factor = None
    return model_args, train_args
