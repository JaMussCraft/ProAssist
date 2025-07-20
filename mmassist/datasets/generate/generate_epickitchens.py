import os
import time
import pandas as pd
from dataclasses import dataclass
from transformers import HfArgumentParser

from dialog_simulation import ParsedVideoAnns
from run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR

IGNORE_VIDEOS = []


@dataclass
class PreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/epic-kitchens"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/epickitchens/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = True


def timestamp_to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_epickitchens_ann(
    ann: dict, max_num_lines_per_gen: int = 20
) -> ParsedVideoAnns:
    video_uid = ann["video_uid"]
    duration = ann["duration"]

    annotated_duration = 0
    num_steps = 0

    for s in ann["steps"]:
        start_time = s["start"]
        end_time = s["end"]
        annotated_duration += end_time - start_time
        num_steps += 1
        time_span = f"[{start_time:.1f}s-{end_time:.1f}s]"
        s["desc"] = f"{time_span} {s['narration']}"
    ann_ratio = annotated_duration / duration

    # get a single string for all the step and substep descriptions
    all_step_descriptions = "\n".join([s["desc"] for s in ann["steps"]])

    # split the descriptions into clips
    clips = []
    clip_st = -1
    additional_ctx_len = 5
    for idx, step in enumerate(ann["steps"]):
        if clip_st < 0:
            clip_st = step["start"]
            clip_start_idx = idx

        if (idx + 1) % max_num_lines_per_gen == 0 or idx == len(ann["steps"]) - 1:
            start_idx = max(clip_start_idx - additional_ctx_len, 0)
            clip_steps = ann["steps"][start_idx : idx + additional_ctx_len]
            clip_description = "\n".join([s["desc"] for s in clip_steps])
            clips.append((clip_st, step["end"], clip_description))
            clip_st = -1

    parsed_ann = ParsedVideoAnns(
        dataset="epickitchens",
        domain="cooking",
        knowedge_type="cooking recipe",
        video_uid=video_uid,
        goal_description="",
        all_step_descriptions=all_step_descriptions,
        clips=clips,
        duration=duration,
        ann_ratio=ann_ratio,
        num_steps=num_steps,
        fps=ann["fps"],
        original_ann=ann,
    )
    return parsed_ann


def load_annotations(args: PreprocessArgs) -> dict[str, list[dict]]:
    ann_dir = os.path.join(args.data_dir, "epic-kitchens-100-annotations")
    video_info_file = os.path.join(ann_dir, "EPIC_100_video_info.csv")
    video_info = pd.read_csv(video_info_file).to_dict(orient="records")

    anns_per_split = {}
    for split in args.splits.split(","):
        ann_split_name = split.replace("val", "validation")
        ann_file = os.path.join(ann_dir, f"EPIC_100_{ann_split_name}.csv")
        action_annotations = pd.read_csv(ann_file).to_dict(orient="records")

        video_id_to_info = {v["video_id"]: v for v in video_info}

        video_annotations = {}
        for ann in action_annotations:
            vid = ann["video_id"]
            vinfo = video_id_to_info[vid]
            if vid not in video_annotations:
                video_annotations[vid] = {
                    "video_uid": vid,
                    "duration": vinfo["duration"],
                    "fps": vinfo["fps"],
                    "steps": [],
                }
            video_ann = video_annotations[vid]
            video_ann["steps"].append(
                {
                    "start": timestamp_to_seconds(ann["start_timestamp"]),
                    "end": timestamp_to_seconds(ann["stop_timestamp"]),
                    "narration": ann["narration"],
                }
            )

        anns_per_split[split] = []
        for vid, ann in video_annotations.items():
            if ann["video_uid"] in IGNORE_VIDEOS:
                continue
            ann["steps"].sort(key=lambda s: s["start"])
            anns_per_split[split].append(ann)

    return anns_per_split


if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArgs, SlurmArguments))
    args, slurm_args = parser.parse_args_into_dataclasses()
    args.tensor_parallel_size = 8 // slurm_args.tasks_per_node

    # load annotations
    anns_per_split = load_annotations(args)

    # submit the job
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args, anns_per_split, parse_epickitchens_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
