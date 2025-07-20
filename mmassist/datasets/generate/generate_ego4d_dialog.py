import os
import time
import json
from dataclasses import dataclass
from transformers import HfArgumentParser

from dialog_simulation import ParsedVideoAnns
from run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR

IGNORE_VIDEOS = [
    "269eea13-c70a-42f6-aba5-41ef622d3112",
    "d2e05761-29c4-4dd5-8ef6-027e40fea282",
]


@dataclass
class PreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/ego4d_track2/v2/annotations"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/ego4d/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = True


def parse_ego4d_goalstep_ann(
    ann: dict, max_num_lines_per_gen: int = 6, essential_only: bool = True
) -> ParsedVideoAnns | None:
    steps = ann["segments"]
    s_idx = 0
    num_substeps = 0
    annotated_duration = 0

    all_descriptions = []
    for ori_sidx, step in enumerate(steps):
        start_time = step["start_time"]
        end_time = step["end_time"]
        annotated_duration += end_time - start_time

        relevance = step.get("is_relevant", "unk")
        if essential_only and relevance != "essential":
            continue

        s_idx += 1

        desc = step["step_description"].lower().replace(".", " ").strip()
        substep_descriptions = ""
        if step["segments"]:
            ss_idx = 0
            substeps = []
            last_substep_desc = ""
            for substep in step["segments"]:
                ss_relevance = step.get("is_relevant", "unk")
                if essential_only and ss_relevance != "essential":
                    continue
                ss_idx += 1
                num_substeps += 1
                sstime = f"[{substep['start_time']:.1f}s]"
                substep_desc = (
                    substep["step_description"].lower().replace(".", " ").strip()
                )
                if substep_desc != last_substep_desc:
                    substeps.append(f" - {sstime} {substep_desc}")
                    last_substep_desc = substep_desc
            substep_descriptions = "\n".join(substeps)
            stime = f"[{step['segments'][0]['start_time']:.1f}s-{end_time:.1f}s]"
        else:
            # stime = f"{step['start_time']:.1f}s-{end_time:.1f}s"
            stime = f"[{start_time:.1f}s-{end_time:.1f}s]"

        # step_description += f"{stime} {desc}\n{substep_descriptions}"
        all_descriptions.append(
            {
                "start": start_time,
                "end": end_time,
                "step": f"{stime} {desc}",
                "substeps": substep_descriptions,
            }
        )

    # get a single string for all the step and substep descriptions
    all_step_descriptions = ""
    for s in all_descriptions:
        all_step_descriptions += s["step"] + "\n"
        if s["substeps"]:
            all_step_descriptions += s["substeps"] + "\n"

    # split the descriptions into clips
    clips = []
    num_lines_in_clip = 0
    clip_start_idx = 0
    clip_start_time = -1
    for idx, step in enumerate(all_descriptions):
        num_lines_in_clip += 1 + step["substeps"].count("\n")

        if clip_start_time < 0:
            clip_start_time = step["start"]

        if (
            num_lines_in_clip > max_num_lines_per_gen
            or idx == len(all_descriptions) - 1
        ):
            # add the clip
            clip_description = ""
            for s_idx, s in enumerate(all_descriptions):
                clip_description += s["step"] + "\n"
                if s_idx > clip_start_idx and s_idx <= idx and s["substeps"]:
                    clip_description += s["substeps"] + "\n"
            clips.append((clip_start_time, step["end"], clip_description))
            clip_start_idx = idx
            clip_start_time = -1
            num_lines_in_clip = 0

    duration = ann["end_time"] - ann["start_time"]
    ann_ratio = annotated_duration / duration

    parsed_ann = ParsedVideoAnns(
        dataset="ego4d",
        domain="cooking",
        knowedge_type="cooking recipe",
        video_uid=ann["video_uid"],
        goal_description=ann["goal_description"],
        all_step_descriptions=all_step_descriptions,
        clips=clips,
        duration=duration,
        ann_ratio=ann_ratio,
        num_steps=s_idx,
        num_substeps=num_substeps,
        original_ann=ann,
    )
    return parsed_ann


def load_annotations(args: PreprocessArgs) -> dict[str, list[dict]]:
    # load data
    anns_per_split = {}
    for split in args.splits.split(","):
        # load data
        ann_file = os.path.join(args.data_dir, f"goalstep_{split}.json")
        with open(ann_file, "r") as f:
            all_anns = json.load(f)["videos"]
        anns_per_split[split] = [
            a
            for a in all_anns
            if a["video_uid"].replace("grp-", "") not in IGNORE_VIDEOS
        ]

    return anns_per_split


if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArgs, SlurmArguments))
    args, slurm_args = parser.parse_args_into_dataclasses()
    args.tensor_parallel_size = 8 // slurm_args.tasks_per_node

    # load annotations
    anns_per_split = load_annotations(args)

    # submit the job
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args, anns_per_split, parse_ego4d_goalstep_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
