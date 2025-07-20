import os
import json
import time
from dataclasses import dataclass
from transformers import HfArgumentParser

from dialog_simulation import ParsedVideoAnns
from run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR

IGNORE_VIDEOS = []


@dataclass
class PreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/holoassist"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/holoassist/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = False


def cleanup_txt(txt: str) -> str:
    return (
        txt.strip()
        .replace("_", " ")
        .replace("student", "user")
        .replace("*unintelligible*", "")
    )


def parse_holoassist_ann(ann: dict, max_num_lines_per_gen: int = 6) -> ParsedVideoAnns:
    video_uid = ann["video_name"]
    task_goal = ann["taskType"]
    duration = ann["videoMetadata"]["duration"]["seconds"]
    fps = ann["videoMetadata"]["video"]["fps"]
    # print(video_uid, task_goal, duration, fps)

    events = ann["events"]
    for e in events:
        if "Conversation Purpose" in e["attributes"]:
            e["start"] = (e["start"] + e["end"]) / 2
    events.sort(key=lambda e: e["start"])

    annotated_duration = 0
    num_steps = 0
    num_substeps = 0
    task_summary = ""
    all_descriptions = []
    substeps = []
    substep_actions = set()

    for e in events:
        start_time = e["start"]
        end_time = e["end"]
        attributes = e["attributes"]

        time_span = f"[{start_time:.1f}s-{end_time:.1f}s]"

        if "Long form description" in attributes:
            task_summary = cleanup_txt(attributes["Long form description"])

        if "Action sentence" in attributes:
            desc = cleanup_txt(attributes["Action sentence"])
            annotated_duration += end_time - start_time
            num_steps += 1
            if all_descriptions:
                all_descriptions[-1]["substeps"] = "\n".join(substeps)
            all_descriptions.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "step": f"{time_span} {desc}",
                    "substeps": "",
                }
            )
            substeps = []
            substep_actions = set()

        elif "Conversation Purpose" in attributes:
            intent = attributes["Conversation Purpose"]
            role = "user" if intent.split("-")[0] == "student" else "assistant"
            utterance = cleanup_txt(attributes["Transcription"])
            if not utterance:
                continue
            num_substeps += 1
            substeps.append(f' - [{start_time:.1f}s] {role}: "{utterance}"')

        elif "Action Correctness" in attributes:
            is_error = attributes["Action Correctness"] != "Correct Action"
            error_reason = cleanup_txt(
                attributes.get("Incorrect Action Explanation", "none")
            )
            verb = cleanup_txt(attributes["Verb"])
            noun = cleanup_txt(attributes["Noun"])
            adj = (
                f"{attributes['Adjective']} "
                if attributes.get("Adjective", "") not in ["none", "wrong", ""]
                else ""
            )
            action = f"{verb} {adj}{noun}"
            err_msg = f" (ERROR: {error_reason})" if is_error else ""

            if verb in ["hold", "touch", "rotate", "inspect"]:
                continue

            if action not in substep_actions:
                num_substeps += 1
                substeps.append(
                    f" - [{start_time:.1f}s-{end_time:.1f}s] {action}{err_msg}"
                )
                substep_actions.add(action)

    # get a single string for all the step and substep descriptions
    all_step_descriptions = ""
    for s in all_descriptions:
        all_step_descriptions += s["step"] + "\n"
        if s["substeps"]:
            all_step_descriptions += s["substeps"] + "\n"

    all_step_descriptions = f"{all_step_descriptions}Summary: {task_summary}\n"

    # split the descriptions into clips
    clips = []
    num_lines_in_clip = 0
    clip_st = -1
    for idx, step in enumerate(all_descriptions):
        num_lines_in_clip += 1 + step["substeps"].count("\n")

        if clip_st < 0:
            clip_st = step["start"]

        if (
            num_lines_in_clip > max_num_lines_per_gen
            or idx == len(all_descriptions) - 1
        ):
            # add the clip
            clip_description = ""
            clip_et = step["end"]
            for s in all_descriptions:
                clip_description += s["step"] + "\n"
                if s["start"] >= clip_st and s["end"] <= clip_et and s["substeps"]:
                    clip_description += s["substeps"] + "\n"
            clip_description += f"Summary: {task_summary}\n"
            clips.append((clip_st, clip_et, clip_description))
            clip_st = -1
            num_lines_in_clip = 0

    ann_ratio = annotated_duration / duration

    parsed_ann = ParsedVideoAnns(
        dataset="holoassist",
        domain="object manipulation",
        knowedge_type="operation manual",
        video_uid=video_uid,
        goal_description=task_goal,
        all_step_descriptions=all_step_descriptions,
        clips=clips,
        duration=duration,
        ann_ratio=ann_ratio,
        num_steps=num_steps,
        num_substeps=num_substeps,
        fps=fps,
        original_ann=ann,
    )
    return parsed_ann


def load_annotations(args: PreprocessArgs) -> dict[str, list[dict]]:
    # load data
    ann_file = os.path.join(args.data_dir, "data-annotation-trainval-v1_1.json")
    with open(ann_file, "r") as f:
        all_annotations: list[dict] = json.load(f)

    anns_per_split = {}
    for split in args.splits.split(","):
        # get the split
        split_vid_file = os.path.join(args.data_dir, f"{split}-v1_2.txt")
        split_vids = set([l.strip() for l in open(split_vid_file, "r")])
        anns_per_split[split] = []
        for ann in all_annotations:
            vid = ann["video_name"]
            ann["video_uid"] = vid
            if vid in IGNORE_VIDEOS:
                continue
            if vid in split_vids:
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
    job = executor.submit(run_jobs, args, anns_per_split, parse_holoassist_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
