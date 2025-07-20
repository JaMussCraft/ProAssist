import os
import json


def get_data_splits(metadata_file: str, output_file: str | None = None) -> dict:
    """Get the data splits for the Ego4D dataset.

    :param metadata_file: path to the metadata file
    :param output_file: path to save the data splits

    :return: a dict of data split name to a list of video UIDs
    """

    print("Loading metadata of all videos...")
    with open(metadata_file, "r") as f:
        all_video_meta = json.load(f)
    all_video_meta = {v["video_uid"]: v for v in all_video_meta["videos"]}

    splits_to_vids = {}
    for video_uid, video_meta in all_video_meta.items():

        splits = ["train"]
        for k in ["split_em", "split_fho", "split_av", "split_goalstep"]:
            # print(k, video_meta[k])
            if video_meta[k] == "test":
                if "train" in splits:
                    splits.remove("train")
                splits.append(f"{k.replace('split_', '')}_test")
            if video_meta[k] == "val":
                if "train" in splits:
                    splits.remove("train")
                splits.append(f"{k.replace('split_', '')}_val")

        for s in splits:
            if s not in splits_to_vids:
                splits_to_vids[s] = set()
            splits_to_vids[s].add(video_uid)

    splits_to_vids = {k: list(v) for k, v in splits_to_vids.items()}

    print("Ego4D data splits:")
    keys = ["train"] + sorted([k for k in splits_to_vids if k != "train"], reverse=True)
    for s in keys:
        print(f"{s}: {len(splits_to_vids[s])} videos")

    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(splits_to_vids, f, indent=2)

    return splits_to_vids


def group_by_split(
    ego4d_data_splits_file: str, our_data_split_file: str, clips: list[dict]
) -> tuple[dict, list]:
    """Group video clips the data split they belong to.

    :param data_splits_file: path to the data splits file
    :param clips: list of video clips where each clip is a dict with "video_uid" key

    :return: a tuple of video_id_to_splits and all_splits
    """
    with open(ego4d_data_splits_file, "r") as f:
        all_splits = json.load(f)
    video_id_to_splits = {}
    for split, video_ids in all_splits.items():
        for vid in video_ids:
            if vid not in video_id_to_splits:
                video_id_to_splits[vid] = []
            video_id_to_splits[vid].append(split)

    with open(our_data_split_file, "r") as f:
        our_valtest_split = json.load(f)

    split_to_videos = {"train": [], "val": [], "test": []}
    for c in clips:
        video_id = c["video_uid"]
        splits = video_id_to_splits[video_id]
        if "train" in splits:
            split_to_videos["train"].append(c)
        elif video_id in our_valtest_split:
            split = our_valtest_split[video_id][0]
            split_to_videos[split].append(c)

    return split_to_videos


def process_narrations(
    narrations: list[str], drop_unsure: bool = True, ego_only: bool = True
) -> str:
    """Process and merge the narrations to a single narration.
    :param narrations: list of narrations
    :param drop_unsure: drop narrations with the "#unsure" tag
    :param ego_only: only keep the ego narration starting with "#C C"

    :return: the merged narration
    """
    assert ego_only, "Only support ego_only mode for now."
    if not narrations:
        return ""

    narration_text = ""
    for idx, t in enumerate(narrations[:2]):
        if drop_unsure and "#unsure" in t.lower():
            continue
        if ego_only and not t.startswith("#C C"):
            continue

        # fix some buggy annotations
        num_cc = t.count("#C C")
        if num_cc > 1:
            t = t[: t.index("#C C", 1)].strip()

        if idx == 0:
            t = t.replace("#C C", "C")
        else:
            t = t.replace("#C C", "and")
        narration_text += t + " "
    narration_text = narration_text.strip()

    if narration_text and not narration_text.endswith("."):
        narration_text += "."
    return narration_text


"""
outdir
    annotations
        <video_id>_<pass_id>.json
        ...
    frames
        <video_id>.arrow
"""


def process_ego4d_narration_data(
    video_id: str,
    video_narr_anns: dict,
    narr_annotator_id: int,
    metadata: dict,
    target_fps: int = 2,
    output_dir: str = "",
) -> dict:

    narr_anns = video_narr_anns[f"narration_pass_{narr_annotator_id}"]["narrations"]
    assert narr_anns, "No narrations found."

    original_fps = metadata["video_metadata"]["fps"]
    num_frames = metadata["video_metadata"]["num_frames"]
    step_size = int((original_fps + 0.5) // target_fps)

    data = {
        "video_uid": video_id,
        "narr_annotator_id": narr_annotator_id,
        "original_fps": original_fps,
        "sample_fps": target_fps,
        "frames_file": f"frames/{video_id}.arrow",
        "conversation": [],
        "raw_narrations": narr_anns,
        "video_meta": metadata,
    }

    num_narr = 0
    ann_pointer = 0
    curr_ann = narr_anns[ann_pointer]
    curr_ann_timestamp = curr_ann["timestamp_frame"]
    sampled_frames_start_idx = 0
    for frame_id in range(0, num_frames, step_size):
        # for frame_file in all_frame_files:
        curr_frame_id = frame_id
        sampled_frame_idx = frame_id // step_size

        texts = []
        while curr_frame_id > curr_ann_timestamp and ann_pointer < len(narr_anns) - 1:
            texts.append(curr_ann["narration_text"])
            ann_pointer += 1
            curr_ann = narr_anns[ann_pointer]
            curr_ann_timestamp = curr_ann["timestamp_frame"]

        text = process_narrations(texts)
        # frame_file_rel = os.path.relpath(os.path.join(frame_out_dir, frame_file), out_dir)
        if text:
            sampled_frames_end_idx = sampled_frame_idx + 1
            data["conversation"].append(
                {
                    "role": "frames",
                    "start": sampled_frames_start_idx,
                    "end": sampled_frames_end_idx,
                }
            )
            data["conversation"].append({"role": "assistant", "content": text})
            sampled_frames_start_idx = sampled_frames_end_idx
            num_narr += 1

        # Note: the last few frames without a following narration will be discarded

        if ann_pointer == len(narr_anns):
            break

    if not data["conversation"]:
        return {}

    # save data
    if output_dir:
        ann_out_dir = os.path.join(output_dir, "processed_narrations")
        os.makedirs(ann_out_dir, exist_ok=True)
        data_file = os.path.join(
            ann_out_dir, f"{video_id}_PASS{narr_annotator_id}.json"
        )
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    return data
