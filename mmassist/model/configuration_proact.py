from enum import Enum
from transformers import PretrainedConfig, LlamaConfig


class ExceedContextHandling(Enum):
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE_AND_DROP = "summarize_and_drop"


class ProActConfig(PretrainedConfig):
    def __init__(
        self,
        *,
        llm_pretrained: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        vision_pretrained: str | None = None,
        img_resolution: int | None = None,
        use_img_cls_token: bool = True,
        img_patch_token_size: int = 0,
        img_patch_token_layer: int = -2,
        img_sep_token: str = "EOS",
        img_token: str = "<image>",
        eos_loss_weight: float = 1.0,
        vision_hidden_size: int = 1152,
        max_seq_len: int = 8192,
        padding_side: str = "left",
        ignore_id: int = -100,
        attn_implementation: str | None = "flash_attention_2",
        w2t_logit_weight: float = 1.0,
        use_binary_decision_head: bool = False,
        binary_loss_weight: float = 1.0,
        exceed_context_handling: str = "drop_all",
        use_pose: bool = False,
        binary_decision_head_type: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_pretrained = llm_pretrained
        self.vision_pretrained = vision_pretrained
        self.img_resolution = img_resolution
        self.use_img_cls_token = use_img_cls_token
        self.img_patch_token_size = img_patch_token_size
        self.img_patch_token_layer = img_patch_token_layer
        self.img_sep_token = img_sep_token
        if img_sep_token == "none":
            self.img_sep_token = ""
        self.img_token = img_token
        self.eos_loss_weight = eos_loss_weight
        self.vision_hidden_size = vision_hidden_size
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side
        self.ignore_id = ignore_id
        self.attn_implementation = attn_implementation
        self.w2t_logit_weight = w2t_logit_weight
        self.use_binary_decision_head = use_binary_decision_head
        self.binary_loss_weight = binary_loss_weight
        self.binary_decision_head_type = binary_decision_head_type
        if exceed_context_handling not in ExceedContextHandling._value2member_map_:
            raise ValueError(
                f"Unsupported exceed_context_handling: {exceed_context_handling}"
            )
        self.exceed_context_handling = exceed_context_handling
        self.use_pose = use_pose

        # add special tokens based on the llm used
        if "Meta-Llama-3" in llm_pretrained:
            if img_sep_token == "EOS":
                self.img_sep_token = "<|eot_id|>"
            self.eos_token = "<|eot_id|>"
            self.chat_formatter_cls = "LLaMA3MultimodalChat"
        else:
            raise ValueError(f"Unsupported LLM model: {llm_pretrained}")

        # update special token ids later after the tokenizer is built
        self.img_token_id = None
        self.img_sep_token_id = None
        self.eos_token_id = None
        self.bor_token_id = None

    @property
    def num_tokens_per_img(self) -> int:
        n = (
            int(self.use_img_cls_token)
            + self.img_patch_token_size * self.img_patch_token_size
        )
        return n

    @property
    def w2t_target_id(self) -> int:
        if self.img_sep_token_id is not None:
            return self.img_sep_token_id
        return self.eos_token_id

    @property
    def exceed_context_handling_stragety(self) -> ExceedContextHandling:
        return ExceedContextHandling(self.exceed_context_handling)


class ProActLlamaConfig(LlamaConfig, ProActConfig):
    @classmethod
    def from_pretrained_llama(
        cls, llm_pretrained: str, **kwargs
    ) -> "ProActLlamaConfig":
        llama_config = LlamaConfig.from_pretrained(llm_pretrained)
        merged_config = llama_config.to_dict()
        merged_config.update(kwargs)
        return cls(llm_pretrained=llm_pretrained, **merged_config)
