import os
import copy
import glob
import pandas as pd
import time
from dataclasses import dataclass
from transformers import HfArgumentParser

from dialog_simulation import ParsedVideoAnns
from run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR


@dataclass
class PreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/assembly101"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/assembly101/generated_dialogs"
    llm: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = False


def parse_assembly101_ann(ann: dict, max_num_lines_per_gen: int = 6) -> ParsedVideoAnns:

    # get a single string for all the step and substep descriptions
    all_step_descriptions = ""
    for s in ann["steps"]:
        mistake = s["mistake"]
        if mistake:
            mistake = f" ({mistake})"
        s_desc = f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['narration']}"
        all_step_descriptions += f"{s_desc}{mistake}\n"
        substeps = s.get("substeps", [])
        if not mistake and len(substeps) > 1:
            for ss in substeps:
                all_step_descriptions += f" - [{ss['start']:.1f}s] {ss['narration']}\n"

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
                mistake = s["mistake"]
                if mistake:
                    mistake = f" ({mistake})"
                s_desc = f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['narration']}"
                clip_description += f"{s_desc}{mistake}\n"
                substeps = s.get("substeps", [])
                if mistake or len(substeps) <= 1:
                    continue
                if s["start"] >= clip_st and s["end"] <= clip_et and s["substeps"]:
                    for ss in substeps:
                        clip_description += (
                            f" - [{ss['start']:.1f}s] {ss['narration']}\n"
                        )
            clips.append((clip_st, clip_et - 1, clip_description))
            clip_st = -1
            num_lines_in_clip = 0

    task_type = "assembly" if ann["task"].startswith("asse") else "disassembly"
    knowedge_type = f"{task_type} steps"
    parsed_ann = ParsedVideoAnns(
        dataset="assembly101",
        domain="assembly/disassembly",
        knowedge_type=knowedge_type,
        video_uid=ann["video_uid"],
        goal_description=ann["task"],
        all_step_descriptions=all_step_descriptions,
        clips=clips,
        duration=ann["duration"],
        ann_ratio=1.0,
        num_steps=len(ann["steps"]),
        num_substeps=sum([len(s["substeps"]) for s in ann["steps"]]),
        original_ann=ann,
        video_start_time=ann["video_start_time"],
        has_mistake=ann["has_mistake"],
    )
    return parsed_ann


def load_annotations(args: PreprocessArgs) -> dict[str, list[dict]]:
    # load data
    data_dir = args.data_dir
    ann_dir = os.path.join(data_dir, "annotations")
    fine_anns_dir = os.path.join(ann_dir, "fine-grained-annotations")
    coarse_anns_dir = os.path.join(ann_dir, "coarse-annotations/coarse_labels")
    mistake_anns_dir = os.path.join(ann_dir, "assembly101-mistake-detection/annots")

    fps = 30  # the frame indexes in the annotations are based on 30 fps
    selected_cam_ids = [
        "HMC_21110305_mono10bit",
        "HMC_21179183_mono10bit",
        "HMC_84355350_mono10bit",
        "HMC_84358933_mono10bit",
    ]  # the best 2 egocentric views
    concrete_actions = ["screw", "unscrew"]  # , "remove", "position"]

    seq_id_to_fine_anns = {}
    for split in ["train", "val"]:
        fine_ann_file = os.path.join(
            fine_anns_dir, f"{split.replace('val', 'validation')}.csv"
        )
        fine_annotations = pd.read_csv(fine_ann_file).to_dict(orient="records")
        for ann in fine_annotations:
            video_file = ann["video"]
            if not any(v in video_file for v in ["HMC_21110305", "HMC_84355350"]):
                continue
            seq_id = video_file.split("/")[0]
            if seq_id not in seq_id_to_fine_anns:
                seq_id_to_fine_anns[seq_id] = {
                    "seq_id": seq_id,
                    "split": split,
                    "substeps": [],
                }
            seq_ann = seq_id_to_fine_anns[seq_id]
            st = ann["start_frame"] / fps
            et = ann["end_frame"] / fps
            seq_ann["substeps"].append(
                {"start": st, "end": et, "narration": ann["action_cls"]}
            )
            seq_ann["toy_name"] = ann["toy_name"] if "-" not in ann["toy_name"] else ""

    coarse_ann_files = glob.glob(os.path.join(coarse_anns_dir, "*.txt"))

    all_seq_ids = set()
    for coarse_ann_file in coarse_ann_files:
        seq_id = coarse_ann_file.split("/")[-1].split("_", 1)[1].split(".")[0]
        all_seq_ids.add(seq_id)

    task_id_to_ann = {}
    for seq_id in all_seq_ids:
        for task_type in ["assembly", "disassembly"]:
            task_id = f"{task_type}_{seq_id}"
            coarse_ann_file = os.path.join(coarse_anns_dir, f"{task_id}.txt")
            if not os.path.exists(coarse_ann_file):
                # print(f"File not found: {coarse_ann_file}")
                continue

            # has coarse annotations
            coarse_ann = pd.read_csv(coarse_ann_file, sep="\t", header=None)
            coarse_ann = coarse_ann.to_dict(orient="records")

            # has fine annotations
            if seq_id not in seq_id_to_fine_anns:
                # print(f"No fine annotations for {seq_id}")
                continue
            fine_anns = seq_id_to_fine_anns[seq_id]

            # has mistake annotations
            mistake_ann_file = os.path.join(mistake_anns_dir, f"{seq_id}.csv")
            if not os.path.exists(mistake_ann_file):
                # print(f"No mistake annotations for {seq_id}")
                continue
            mistake_anns = pd.read_csv(mistake_ann_file, header=None)
            mistake_anns = mistake_anns.to_dict(orient="records")

            fine_anns = seq_id_to_fine_anns[seq_id]
            toy_name = (
                "toy " + fine_anns["toy_name"] if fine_anns["toy_name"] else "toy"
            )
            task_name = f"{task_type.replace('ly', 'le')} {toy_name}"

            if task_id not in task_id_to_ann:
                task_id_to_ann[task_id] = copy.deepcopy(fine_anns)
                task_id_to_ann[task_id].update(
                    {"task_id": task_id, "task": task_name, "steps": []}
                )

            has_mistake = False
            ann = task_id_to_ann[task_id]
            for c in coarse_ann:
                narration = c[2]
                mistake_msg = ""
                for m in mistake_anns:
                    if m[0] == c[0]:
                        prop = "to" if m[2] == "attach" else "from"
                        narration = f"{m[2]} {m[3]} {prop} {m[4]}"
                        if m[5] == "mistake":
                            mistake_msg = f"mistake: {m[6]}"
                            has_mistake = True
                        elif m[5] == "correction":
                            if m[2] == "attach":
                                action_wrong = narration.replace("attach", "detach")
                                action_wrong = action_wrong.replace("to", "from")
                                mistake_msg = f'correction of "{action_wrong}"'
                            else:
                                action_wrong = narration.replace("detach", "attach")
                                action_wrong = action_wrong.replace("from", "to")
                                mistake_msg = f'correction of "{action_wrong}"'
                        break
                st, et = c[0] / fps, c[1] / fps
                ann["steps"].append(
                    {
                        "start": st,
                        "end": et,
                        "narration": narration,
                        "mistake": mistake_msg,
                        "substeps": [],
                    }
                )
                for s in fine_anns["substeps"]:
                    add = False
                    for verb in concrete_actions:
                        if s["narration"].startswith(verb):
                            add = True
                            break
                    if add and s["start"] >= st and s["start"] <= et:
                        ann["steps"][-1]["substeps"].append(s)
                ann["steps"][-1]["substeps"].sort(key=lambda s: s["start"])
                ann["has_mistake"] = has_mistake

    for task_id, ann in task_id_to_ann.items():
        ann["steps"].sort(key=lambda s: s["start"])
        ann["substeps"].sort(key=lambda s: s["start"])

        start_time = ann["steps"][0]["start"]
        duration = ann["steps"][-1]["end"] - start_time
        ann["video_start_time"] = max(0, start_time - 2)
        ann["duration"] = duration

    video_dir = os.path.join(data_dir, "videos")
    video_uid_to_ann = {}
    for task_id, ann in task_id_to_ann.items():
        seq_id = ann["seq_id"]
        for cam_id in selected_cam_ids:
            video_file = os.path.join(video_dir, f"{seq_id}/{cam_id}.mp4")
            if os.path.exists(video_file):
                video_uid = f"{task_id}__{cam_id}"
                ann = copy.deepcopy(ann)
                ann["video_uid"] = video_uid
                video_uid_to_ann[video_uid] = ann

    all_annotations = [v for v in video_uid_to_ann.values()]
    all_annotations.sort(key=lambda v: v["video_uid"])

    anns_per_split = {}
    for split in args.splits.split(","):
        anns_per_split[split] = [v for v in all_annotations if v["split"] == split]

    return anns_per_split


if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArgs, SlurmArguments))
    args, slurm_args = parser.parse_args_into_dataclasses()
    args.tensor_parallel_size = 8 // slurm_args.tasks_per_node

    # load annotations
    anns_per_split = load_annotations(args)

    # submit the job
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args, anns_per_split, parse_assembly101_ann)

    # gather and save results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
