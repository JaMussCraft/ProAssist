import json
import random
import pyarrow
import pyarrow.json
import warnings
import time
import base64
import torch
from PIL import Image

from torchvision.io import image as torch_imageio
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import to_pil_image, normalize

warnings.filterwarnings("ignore")


class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self


def inverse_preprocess_to_pil_images(frames: torch.Tensor, mean: list, std: list):
    frames = normalize(
        frames,
        mean=tuple(-m / s for m, s in zip(mean, std)),
        std=tuple(1.0 / s for s in std),
    )
    frames = (frames * 255).to(torch.uint8)
    return list(map(to_pil_image, frames))


def rand_bool():
    return bool(random.getrandbits(1))


def case_connect(prefix: str, suffix: str):
    if not prefix:
        return suffix[0].upper() + suffix[1:]
    if not suffix:
        return prefix
    if prefix[-1] == "," or prefix[-1] == ":":
        return prefix + " " + suffix[0].lower() + suffix[1:]
    return prefix + " " + suffix[0].upper() + suffix[1:]


def batch_temporal_iou(sequences1: torch.Tensor, sequences2: torch.Tensor):
    area1 = sequences1[:, 1] - sequences1[:, 0]
    area2 = sequences2[:, 1] - sequences2[:, 0]
    l = torch.maximum(sequences1[:, None, 0], sequences2[:, 0])
    r = torch.minimum(sequences1[:, None, 1], sequences2[:, 1])
    inter = (r - l).clamp(min=0)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def temporal_iou(region1, region2):
    area1 = region1[1] - region1[0]
    area2 = region2[1] - region2[0]
    l = max(region1[0], region2[0])
    r = min(region1[1], region2[1])
    inter = max(0, (r - l))
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def load_jsonl(file_path: str):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def load_jsonl_to_arrow(file_path: str, verbose: bool = False) -> pyarrow.Table:
    st = time.time()
    data = pyarrow.json.read_json(file_path)
    if verbose:
        print(f"File: {file_path} loaded in {time.time() - st: .2f} seconds.")
    return data


def img_base64_to_tensor(img_base64: str) -> torch.Tensor:
    img_bytes = torch.frombuffer(base64.b64decode(img_base64), dtype=torch.uint8)
    img = torch_imageio.decode_image(img_bytes, mode=ImageReadMode.RGB)
    return img


def format_size(size: int) -> str:
    """Convert a size to human readable format."""
    magnitude = 0
    while abs(size) >= 1000:
        magnitude += 1
        size /= 1000.0
    if magnitude == 0:
        return "%d" % size
    return "%.1f%s" % (size, ["", "K", "M", "G", "T", "P"][magnitude])


def tensor_to_pil_images(tensors: torch.Tensor) -> list[Image.Image]:
    """Convert a list of tensor images to PIL images.

    :param tensors: list of tensor images of shape (B, C, H, W)
    :return: list of PIL images
    """
    return [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in tensors]


def sample_frames(frames: torch.Tensor, N: int = 5) -> torch.Tensor:
    T = len(frames)
    indices = torch.randperm(T)[:N]
    sorted_indices = torch.sort(indices).values
    return frames[sorted_indices]
