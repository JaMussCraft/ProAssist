import math, torch
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel, AutoProcessor

from .configuration_proact import ProActConfig


def adaptive_avg_pool2d(patch_tokens: Tensor, pooled_size: tuple[int, int]) -> Tensor:
    """Adaptive average pooling for 2D tensors.

    :param patch_tokens: patch feature map of shape BxLxC
    :param pooled_size: the target 2d spatial size of the pooled tensor
    :return: the pooled tensor of shape Bx(pooled_size[0]*pooled_size[1])xC
    """
    s = int(math.sqrt(patch_tokens.shape[1]))
    patch_tokens = patch_tokens.reshape(
        patch_tokens.shape[0], s, s, patch_tokens.shape[-1]
    ).permute(0, 3, 1, 2)
    patch_tokens = (
        torch.nn.functional.adaptive_avg_pool2d(
            patch_tokens,
            pooled_size,
        )
        .flatten(2, 3)
        .permute(0, 2, 1)
    )
    return patch_tokens


def clip_vision_encode(
    model: nn.Module,
    frames: Tensor,
    use_cls_token: bool,
    patch_tokens_size: tuple[int, int] | None,
    patch_token_layer: int = -1,
    do_preprocess: bool = True,
    input_size: tuple[int, int] = (384, 384),
    mean: list[float] = [0.5, 0.5, 0.5],
    std: list[float] = [0.5, 0.5, 0.5],
    rescale_factor: float = 0.00392156862745098,  # 1/255
    patch_token_start_idx: int = 1,
    **kwargs,
) -> Tensor:
    if do_preprocess:
        frames = normalize(frames * rescale_factor, mean=mean, std=std)

        # resize to input_size
        if frames.shape[-2:] != input_size:
            frames = nn.functional.interpolate(frames, size=input_size, mode="bilinear")

    output_hidden_states = patch_token_layer != -1
    vision_outputs = model(frames, output_hidden_states=output_hidden_states)

    outputs = []
    if use_cls_token:
        cls_token = vision_outputs.pooler_output[:, None]  # Bx1xC
        outputs.append(cls_token)

    if patch_tokens_size is not None:
        if patch_token_layer == -1:
            last_hidden_state = vision_outputs.last_hidden_state
            patch_tokens = last_hidden_state[:, patch_token_start_idx:]  # BxLxC
        else:
            hidden_states = vision_outputs.hidden_states
            patch_tokens = hidden_states[patch_token_layer][:, patch_token_start_idx:]
        if patch_tokens_size[0] != -1:
            patch_tokens = adaptive_avg_pool2d(patch_tokens, patch_tokens_size)
        outputs.append(patch_tokens)

    return torch.cat(outputs, dim=1)  # BxLxC


class VisualEncoder(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        processor: AutoProcessor,
        use_cls_token: bool,
        patch_tokens_size: tuple[int, int] | None,
        patch_token_layer: int = -1,
        input_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vision_model = model.vision_model
        self.processor = processor
        self.use_cls_token = use_cls_token
        self.patch_tokens_size = patch_tokens_size
        self.patch_token_layer = patch_token_layer
        self.mean = processor.image_processor.image_mean
        self.std = processor.image_processor.image_std
        self.rescale_factor = processor.image_processor.rescale_factor
        self.patch_token_start_idx = 0 if "siglip" in model.name_or_path else 1
        if input_size is not None:
            self.input_size = input_size
        elif hasattr(processor.image_processor, "size"):
            _input_size = processor.image_processor.size
            if "width" in _input_size and "height" in _input_size:
                self.input_size = (_input_size["height"], _input_size["width"])
            elif "shortest_edge" in _input_size:
                self.input_size = (
                    _input_size["shortest_edge"],
                    _input_size["shortest_edge"],
                )
            else:
                raise ValueError(
                    "Cannot get the input image size. Please specify it manually."
                )
        else:
            raise ValueError("Please specify the input image size")

    @property
    def device(self) -> torch.device:
        return self.vision_model.embeddings.patch_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    @classmethod
    def from_config(cls, config: ProActConfig) -> "VisualEncoder":
        pts = config.img_patch_token_size
        return cls(
            model=AutoModel.from_pretrained(config.vision_pretrained),
            processor=AutoProcessor.from_pretrained(config.vision_pretrained),
            use_cls_token=config.use_img_cls_token,
            patch_tokens_size=((pts, pts) if pts != 0 else None),
            input_size=(
                (config.img_resolution, config.img_resolution)
                if config.img_resolution is not None
                else None
            ),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(device=self.device, dtype=self.dtype)
        return clip_vision_encode(
            model=self.vision_model,
            frames=frames,
            use_cls_token=self.use_cls_token,
            patch_tokens_size=self.patch_tokens_size,
            input_size=self.input_size,
            mean=self.mean,
            std=self.std,
            rescale_f=self.rescale_factor,
            patch_token_start_idx=self.patch_token_start_idx,
        )
