import os
import logging
from torch.utils.data import ConcatDataset, Dataset

from mmassist.data.dataset import BaseDataset
from mmassist.data.utils import format_size


def parse_dataset_name(dataset_name: str, data_root_dir: str) -> tuple[str, str]:
    """Parse the dataset name to get the data folder and annotation file.
    Note: have to use the same convention in data preparation

    :param dataset_name: dataset name in the format of {data_folder}/{annotation_file}
    :param data_root_dir: the root directory of all the data

    :return: a tuple of data folder and annotation file
    """
    data_folder, ann_file_name = dataset_name.split("/")

    data_dir = os.path.join(data_root_dir, data_folder)
    ann_file = os.path.join(data_dir, "prepared", f"{ann_file_name}.jsonl")

    return data_dir, ann_file


def get_feature_dir_and_keys(
    root_dir: str,
    vision_pretrained: str | None = None,
    use_img_cls_token: bool = True,
    img_patch_token_size: int = 0,
    img_patch_token_layer: int = -2,
    **kwargs,
) -> tuple[str | None, list[str]]:
    """Get the image feature directory and keys.
    Note: have to use the same convention in data preparation

    :param vision_pretrained: the vision model pretrained name
    :param use_img_cls_token: whether to use the [CLS] token from CLIP
    :param img_patch_token_size: the size of patch tokens map to use from CLIP
    :param img_patch_token_layer: the layer to extract patch tokens from CLIP

    :return: a tuple of image feature directory and keys
    """

    img_feature_dir = None
    if vision_pretrained is not None:
        model_name = vision_pretrained.replace("/", "___")
        rel_dir = f"features/{model_name}@{img_patch_token_layer}"
        img_feature_dir = os.path.join(root_dir, rel_dir)

    img_feature_keys = []
    if use_img_cls_token:
        img_feature_keys.append("cls")
    if img_patch_token_size != 0:
        img_feature_keys.append(f"{img_patch_token_size}x{img_patch_token_size}")

    return img_feature_dir, img_feature_keys


def build_dataset(dataset_name: str, data_root_dir: str, **kwargs) -> Dataset:
    data_dir, ann_file = parse_dataset_name(dataset_name, data_root_dir)
    img_f_dir, img_f_keys = get_feature_dir_and_keys(data_dir, **kwargs)
    return BaseDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        ann_file=ann_file,
        img_feature_dir=img_f_dir,
        img_feature_keys=img_f_keys,
        **kwargs,
    )


def get_dataset_shortname(dataset_name: str) -> str:
    """Get the split name from the dataset name."""
    if "val" in dataset_name:
        return dataset_name.split("val")[0] + "val"
    elif "test" in dataset_name:
        return dataset_name.split("test")[0] + "test"
    return dataset_name


def build_train_dataset(
    train_datasets: str | list[str], print_info: bool = False, **kwargs
) -> ConcatDataset:
    """Build the training datasets.

    :param train_datasets: dataset names to use, separated by comma

    :return: a ConcatDataset of the training datasets
    """
    if isinstance(train_datasets, str):
        train_datasets = train_datasets.split(",")

    assert train_datasets, "No training datasets provided."

    train_data = []
    total_train_size = 0
    log_str = "Training datasets:\n"
    for dataset_name in train_datasets:
        if "@" in dataset_name:
            dataset_name, repeat = dataset_name.split("@")
            repeat = int(repeat)
        else:
            repeat = 1
        dataset = build_dataset(dataset_name, repeat=repeat, **kwargs)
        train_data.append(dataset)
        data_size = len(dataset)
        total_train_size += data_size
        log_str += f"* {dataset_name}@{repeat} | num samples: {data_size}\n"

    log_str += f"Total training data size: {format_size(total_train_size)}"
    if print_info:
        print(log_str)

    return ConcatDataset(train_data)


def build_eval_datasets(
    eval_datasets: str | list[str] | None, print_info: bool = False, **kwargs
) -> dict[str, Dataset]:
    """Build the evaluation datasets.

    :param eval_datasets: dataset names to use, separated by comma

    :return: a dict of datasets
    """
    if eval_datasets is None:
        return {}
    if isinstance(eval_datasets, str):
        eval_datasets = eval_datasets.split(",")

    assert eval_datasets, "No evaluation datasets provided."

    datasets = {}
    log_str = "Evaluation datasets:\n"
    for dataset_name in eval_datasets:
        dataset = build_dataset(dataset_name, **kwargs)
        short_name = get_dataset_shortname(dataset_name)
        datasets[short_name] = dataset
        log_str += f"* {short_name} | num samples: {len(dataset)}\n"

    if print_info:
        print(log_str)

    return datasets
