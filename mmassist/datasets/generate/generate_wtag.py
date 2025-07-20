import os
import json
import time
from dataclasses import dataclass
from transformers import HfArgumentParser

from dialog_simulation import ParsedVideoAnns
from run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR


@dataclass
class PreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/WTaG"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/wtag/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.3
    filter_by_llm: bool = False


def parse_wtag_ann(ann: dict, max_num_lines_per_gen: int = 6) -> ParsedVideoAnns:
    video_uid = ann["video_uid"]
    duration = ann["duration"]
    ann_ratio = ann["ann_ratio"]
    num_steps = len(ann["steps"])

    # get a single string for all the step and substep descriptions
    all_step_descriptions = ""
    for s in ann["steps"]:
        all_step_descriptions += s["step"] + "\n"
        if s.get("substeps"):
            all_step_descriptions += "\n".join(s["substeps"]) + "\n"

    # split the descriptions into clips
    clips = []
    num_lines_in_clip = 0
    clip_st = -1
    for idx, step in enumerate(ann["steps"]):
        num_lines_in_clip += 1 + step["substeps"].count("\n")

        if clip_st < 0:
            clip_st = step["start"]

        if num_lines_in_clip > max_num_lines_per_gen or idx == len(ann["steps"]) - 1:
            # add the clip
            clip_description = ""
            clip_et = step["end"]
            for s in ann["steps"]:
                clip_description += s["step"] + "\n"
                if s["start"] >= clip_st and s["end"] <= clip_et and s["substeps"]:
                    clip_description += "\n".join(s["substeps"]) + "\n"
            clips.append((clip_st, clip_et, clip_description))
            clip_st = -1
            num_lines_in_clip = 0

    parsed_ann = ParsedVideoAnns(
        dataset="wtag",
        domain="cooking",
        knowedge_type="cooking recipe",
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
    video_ids = sorted([f for f in os.listdir(args.data_dir)])
    video_ids.remove("T55a")
    video_ids.remove("T55b")

    sampling_rate = 10_000_000

    all_annotations = []
    for vid in video_ids:
        timing_file = os.path.join(args.data_dir, vid, "Video/VideoMpegTiming.txt")
        with open(timing_file, "r") as f:
            lines = f.readlines()
            start_ts = float(lines[0].strip())
            end_ts = float(lines[1].strip())
            duration = (end_ts - start_ts) / sampling_rate

        # add step annotations
        events = []
        for ann_file, role in [
            ("StepDetection/StepDetection.txt", ""),
            ("TextASR/InstructorAnnotations_intent.txt", "assistant"),
            ("TextASR/UserAnnotations_intent.txt", "user"),
        ]:
            anns = []
            with open(os.path.join(args.data_dir, vid, ann_file), "r") as f:
                for line in f:
                    anns.append(line.strip().split("\t"))
            for ann in anns:
                st = (int(ann[0]) - start_ts) / sampling_rate
                et = (int(ann[1]) - start_ts) / sampling_rate
                if "TextASR" in ann_file:
                    st = (st + et) / 2  # use the middle time for dialog annotations
                content = ann[2] if not role else f'{role}: "{ann[2]}"'
                events.append({"start": st, "end": et, "narration": content})
        events.sort(key=lambda s: s["start"])

        end_of_start_step = 0
        for e in events:
            if e["narration"].lower() == "start":
                end_of_start_step = e["end"]
                break

        steps = []
        ann_duration = 0
        for e in events:
            if e["start"] < end_of_start_step:
                continue
            if ":" not in e["narration"]:
                e["step"] = f"[{e['start']:.1f}s-{e['end']:.1f}s] {e['narration']}"
                e["substeps"] = []
                ann_duration += e["end"] - e["start"]
                steps.append(e)
            elif steps:
                last_step = steps[-1]
                ss_desc = f"- [{e['start']:.1f}s] {e['narration']}"
                last_step["substeps"].append(ss_desc)
                if e["start"] > last_step["end"]:
                    last_step["end"] = e["start"]
                    last_step["step"] = (
                        f"[{last_step['start']:.1f}s-{last_step['end']:.1f}s] {last_step['narration']}"
                    )
        dummy_start_time = end_of_start_step / 2
        steps.insert(
            0,
            {
                "start": dummy_start_time,
                "end": end_of_start_step,
                "step": f"- [{dummy_start_time:.1f}s] start",
                "substeps": [],
            },
        )
        ann_ratio = ann_duration / duration
        video_ann = {
            "video_uid": vid,
            "duration": duration,
            "ann_ratio": ann_ratio,
            "steps": steps,
        }
        all_annotations.append(video_ann)

    anns_per_split = {}
    for split in args.splits.split(","):
        if split == "train":
            anns_per_split[split] = all_annotations[:40]
        elif split == "val":
            anns_per_split[split] = all_annotations[40:]
        else:
            raise ValueError(f"Unknown split {split}")

    return anns_per_split


if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArgs, SlurmArguments))
    args, slurm_args = parser.parse_args_into_dataclasses()
    args.tensor_parallel_size = 8 // slurm_args.tasks_per_node

    # load annotations
    anns_per_split = load_annotations(args)

    # submit the job
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args, anns_per_split, parse_wtag_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
