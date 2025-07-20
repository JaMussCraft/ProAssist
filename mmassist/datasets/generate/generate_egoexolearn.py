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
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/EgoExoLearn"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/egoexolearn/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.4
    filter_by_llm: bool = False


def parse_egoexolearn_ann(
    ann: dict, max_num_lines_per_gen: int = 10
) -> ParsedVideoAnns:
    video_uid = ann["video_uid"]
    duration = ann["steps"][-1]["end"] - ann["steps"][0]["start"]

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
    for idx, step in enumerate(ann["steps"]):
        if clip_st < 0:
            clip_st = step["start"]
            clip_start_idx = idx

        if (idx + 1) % max_num_lines_per_gen == 0 or idx == len(ann["steps"]) - 1:
            clip_steps = ann["steps"][max(clip_start_idx - 5, 0) : idx + 5]
            clip_description = "\n".join([s["desc"] for s in clip_steps])
            clips.append((clip_st, step["end"], clip_description))
            clip_st = -1

    domain = "cooking" if ann["scene"] == "kitchen" else "lab task"
    ktype = "cooking recipe" if domain == "cooking" else "lab task steps"
    parsed_ann = ParsedVideoAnns(
        dataset="egoexolearn",
        domain=domain,
        knowedge_type=ktype,
        video_uid=video_uid,
        goal_description="",
        all_step_descriptions=all_step_descriptions,
        clips=clips,
        duration=duration,
        ann_ratio=ann_ratio,
        num_steps=num_steps,
        original_ann=ann,
    )
    return parsed_ann


def load_annotations(args: PreprocessArgs) -> dict[str, list[dict]]:
    # load data
    ann_dir = os.path.join(args.data_dir, "annotations")
    ann_file = os.path.join(ann_dir, "fine_annotation_trainval_en.csv")
    all_step_annotations = pd.read_csv(ann_file).to_dict(orient="records")

    all_annotations = {}
    for ann in all_step_annotations:
        if ann["view"] != "ego":
            continue
        vid = ann["video_uid"]
        if vid not in all_annotations:
            all_annotations[vid] = {
                "video_uid": vid,
                "split": ann["subset"],
                "scene": ann["scene"],
                "steps": [],
            }
        video_ann = all_annotations[vid]
        video_ann["steps"].append(
            {
                "start": ann["start_sec"],
                "end": ann["end_sec"],
                "narration": ann["narration_en_no_hand_prompt"],
            }
        )
    ## sort the steps by start time
    for vid, ann in all_annotations.items():
        ann["steps"].sort(key=lambda s: s["start"])

    anns_per_split = {}
    for split in args.splits.split(","):
        # get the split
        anns_per_split[split] = []
        for ann in all_annotations.values():
            if ann["video_uid"] in IGNORE_VIDEOS:
                continue
            if ann["split"] == split:
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
    job = executor.submit(run_jobs, args, anns_per_split, parse_egoexolearn_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
