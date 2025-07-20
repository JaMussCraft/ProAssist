import os
import json


def get_file_path(base_dir: str, sample_idx: int, file_ext: str = "json") -> str:
    """Get the file path for the sample of the given index."""
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{sample_idx}.{file_ext}")


def save_json(data: dict, file_path: str, indent: int | None = 2) -> None:
    """Save a dictionary to a json file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def parse_inference_setups(inference_setups: str) -> dict[str, dict[str, list[dict]]]:
    """Parse the inference configurations."""
    setups = {}
    for setup_str in inference_setups.split(","):
        (
            dataset,
            runner,
            eval_max_seq_len_str,
            not_talk_thresh,
            ctx_handling_method,
        ) = setup_str.split("|")
        if dataset not in setups:
            setups[dataset] = {}
        if runner not in setups[dataset]:
            setups[dataset][runner] = []

        eval_max_seq_len = int(eval_max_seq_len_str[:-1]) * 1024
        setup = {
            "inference_runner_type": runner,
            "eval_max_seq_len_str": eval_max_seq_len_str,
            "eval_max_seq_len": eval_max_seq_len,
            "not_talk_threshold": float(not_talk_thresh),
        }
        if runner == "stream":
            setup["context_handling_method"] = ctx_handling_method
        setups[dataset][runner].append(setup)

    return setups


def get_match_time_window(dataset: str) -> tuple[float, float]:
    """Get the match time window for the given dataset.

    The match time [left, right] means a predicted utterance can match with the
    a reference utterance if the reference time is in the range of
    [predicted_time + left, predicted_time + right] seconds.

    Match time obtained from the average talking interval
    in each dataset as follow:
    - For narration data: (-talk_interval / 2, talk_interval / 2)
    - For assistant dialog: (-talk_interval / 4, talk_interval / 2)
    where we apply a smaller left bound for the assistant dialog data because
    we do not want the predicted assistant utterance to reflect something
    has already happened.
    """
    if "narration" in dataset:
        return (-2.5, 2.5)
    elif "ego4d" in dataset:
        return (-4.0, 8.5)
    elif "epickitchens" in dataset:
        return (-1.5, 3.0)
    elif "holoassist" in dataset:
        return (-1.5, 2.5)
    elif "egoexolearn" in dataset:
        return (-2.0, 3.5)
    elif "assembly101" in dataset:
        return (-2.0, 4.0)
    elif "wtag" in dataset:
        return (-3.0, 6.0)
    return (-2.5, 2.5)
