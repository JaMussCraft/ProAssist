import os
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import HfArgumentParser

from mmassist.datasets.generate.dialog_simulation import ParsedVideoAnns
# from mmassist.datasets.generate.run_utils import run_jobs, get_slurm_executor, save_results, SlurmArguments
from mmassist.configs.arguments import DATA_ROOT_DIR


@dataclass
class EpflPreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/epfl/annotations"
    frames_dir: str = f"{DATA_ROOT_DIR}/processed_data/epfl/frames"
    splits: str = "train,test"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/epfl/generated_dialogs"
    llm: str = "anthropic/claude-3.5-sonnet"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.3  # Lower threshold for EPFL since it has sparser annotations
    filter_by_llm: bool = True
    max_num_lines_per_gen: int = 20  # As specified in requirements


def load_epfl_annotations(annotation_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load both coarse and fine-grained annotations for a single session.
    
    Args:
        annotation_dir: Path to the annotations directory for a specific session
        
    Returns:
        Tuple of (coarse_annotations, fine_grained_annotations)
    """
    # Load coarse annotations
    activity_file = os.path.join(annotation_dir, "activity_annotations.json")
    with open(activity_file, 'r') as f:
        activity_data = json.load(f)
    coarse_annotations = activity_data["annotations"]
    
    # Load fine-grained annotations
    actions_file = os.path.join(annotation_dir, "actions_annotations.xlsx")
    actions_df = pd.read_excel(actions_file)
    
    # Filter out confused annotations (confusion=1)
    actions_df = actions_df[actions_df['Confusion'] != 1]
    
    # Combine verbs and nouns into action descriptions
    fine_grained_annotations = []
    for _, row in actions_df.iterrows():
        verb = str(row['Verbs']) if pd.notna(row['Verbs']) else ""
        noun = str(row['Nouns']) if pd.notna(row['Nouns']) else ""
        
        # Create action description by combining verb and noun
        if verb and noun:
            action_desc = f"{verb} {noun}".lower().strip()
        elif verb:
            action_desc = verb.lower().strip()
        else:
            continue  # Skip empty actions
        
        fine_grained_annotations.append({
            "start": float(row['Start']),
            "end": float(row['End']),
            "action": action_desc
        })
    
    return coarse_annotations, fine_grained_annotations


def combine_annotations(coarse_annotations: List[Dict], fine_grained_annotations: List[Dict]) -> List[Dict]:
    """
    Combine coarse and fine-grained annotations by grouping fine-grained actions
    under coarse activities based on start time.
    
    Args:
        coarse_annotations: List of coarse activity annotations
        fine_grained_annotations: List of fine-grained action annotations
        
    Returns:
        List of combined annotations with fine-grained actions grouped under coarse activities
    """
    combined = []
    
    for coarse_ann in coarse_annotations:
        coarse_start = coarse_ann["start"]
        coarse_end = coarse_ann["end"]
        
        # Find all fine-grained actions that fall within this coarse activity
        actions_in_activity = []
        for fine_ann in fine_grained_annotations:
            # Check if fine-grained action overlaps with coarse activity
            if (fine_ann["start"] >= coarse_start and fine_ann["start"] < coarse_end) or \
               (fine_ann["end"] > coarse_start and fine_ann["end"] <= coarse_end) or \
               (fine_ann["start"] < coarse_start and fine_ann["end"] > coarse_end):
                actions_in_activity.append(fine_ann)
        
        combined.append({
            "start": coarse_start,
            "end": coarse_end,
            "activity": coarse_ann["Activities"],
            "actions": actions_in_activity
        })
    
    return combined


def parse_epfl_annotations(
    split: str,
    participant: str,
    session: str, 
    annotation_dir: str,
    max_num_lines_per_gen: int = 20
) -> Optional[ParsedVideoAnns]:
    """
    Parse EPFL annotations into the format expected by the dialog generation pipeline.
    
    Args:
        participant: Participant ID (e.g., 'YH2003')
        session: Session ID (e.g., '2023_05_17_09_08_58')
        annotation_dir: Path to the annotations directory
        frames_dir: Path to the frames directory
        max_num_lines_per_gen: Maximum number of lines per generation clip
        
    Returns:
        ParsedVideoAnns object or None if parsing fails
    """
    try:
        # Load annotations
        coarse_annotations, fine_grained_annotations = load_epfl_annotations(annotation_dir)
        
        # Combine annotations
        combined_annotations = combine_annotations(coarse_annotations, fine_grained_annotations)
        
        # Create step descriptions
        all_descriptions = []
        for ann in combined_annotations:
            start_time = ann["start"]
            end_time = ann["end"]
            activity = ann["activity"]
            
            # Create time-stamped activity description
            activity_desc = f"[{start_time:.1f}s-{end_time:.1f}s] {activity}"
            
            # Add fine-grained actions as substeps
            substeps = []
            for action in ann["actions"]:
                action_time = f"[{action['start']:.1f}s]"
                substeps.append(f" - {action_time} {action['action']}")
            
            all_descriptions.append({
                "start": start_time,
                "end": end_time,
                "step": activity_desc,
                "substeps": "\n".join(substeps) if substeps else ""
            })
        
        # Create all step descriptions string
        all_step_descriptions = ""
        for desc in all_descriptions:
            all_step_descriptions += desc["step"] + "\n"
            if desc["substeps"]:
                all_step_descriptions += desc["substeps"] + "\n"
        
        # Split into clips based on max_num_lines_per_gen
        clips = []
        num_lines_in_clip = 0
        clip_start_idx = 0
        clip_start_time = -1
        
        for idx, desc in enumerate(all_descriptions):
            num_lines_in_clip += 1
            
            if clip_start_time < 0:
                clip_start_time = desc["start"]
            
            if (num_lines_in_clip > max_num_lines_per_gen or 
                idx == len(all_descriptions) - 1):
                
                # Create clip description
                clip_description = ""
                for s_idx in range(clip_start_idx, idx + 1):
                    s = all_descriptions[s_idx]
                    clip_description += s["step"] + "\n"
                    if s["substeps"]:
                        clip_description += s["substeps"] + "\n"
                
                clips.append((clip_start_time, desc["end"], clip_description.strip()))
                clip_start_idx = idx + 1
                clip_start_time = -1
                num_lines_in_clip = 0
        
        # Calculate metrics
        total_duration = combined_annotations[-1]["end"]
        # total_duration = max(ann["end"] for ann in combined_annotations) - min(ann["start"] for ann in combined_annotations)
        annotated_duration = sum(ann["end"] - ann["start"] for ann in combined_annotations)
        ann_ratio = annotated_duration / total_duration if total_duration > 0 else 0
                
        # Create video UID
        video_uid = f"{split}_{participant}_{session}"
        
        parsed_ann = ParsedVideoAnns(
            dataset="epfl",
            domain="cooking",
            knowledge_type="cooking recipe",
            video_uid=video_uid,
            goal_description="",
            all_step_descriptions=all_step_descriptions,
            clips=clips,
            duration=total_duration,
            ann_ratio=ann_ratio,
            num_steps=len(combined_annotations),
            num_substeps=sum(len(ann["actions"]) for ann in combined_annotations),
            original_ann={
                "participant": participant,
                "session": session,
                "coarse_annotations": coarse_annotations,
                "fine_grained_annotations": fine_grained_annotations,
                "combined_annotations": combined_annotations
            }
        )
        
        return parsed_ann
        
    except Exception as e:
        print(f"Error parsing annotations for {participant}/{session}: {e}")
        return None


def parse_epfl_annotation_wrapper(
    ann: dict, max_num_lines_per_gen: int = 20
) -> Optional[ParsedVideoAnns]:
    """
    Wrapper function to match the expected signature for run_jobs.
    
    Args:
        ann: Dictionary containing participant, session, and annotation directory info
        max_num_lines_per_gen: Maximum number of lines per generation clip
        
    Returns:
        ParsedVideoAnns object or None if parsing fails
    """
    split = ann["split"]
    participant = ann["participant"]
    session = ann["session"]
    annotation_dir = ann["annotation_dir"]
    
    return parse_epfl_annotations(
        split, participant, session, annotation_dir, max_num_lines_per_gen
    )


def load_epfl_dataset(args: EpflPreprocessArgs) -> Dict[str, List[ParsedVideoAnns]]:
    """
    Load all EPFL annotations for the specified splits.
    
    Args:
        args: Preprocessing arguments
        
    Returns:
        Dictionary mapping split names to lists of ParsedVideoAnns
    """
    anns_per_split = {}
    
    for split in args.splits.split(","):
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        
        split_annotations = []
        
        # Iterate through participants
        for participant in os.listdir(split_dir):
            participant_dir = os.path.join(split_dir, participant)
            if not os.path.isdir(participant_dir):
                continue
            
            # Iterate through sessions
            for session in os.listdir(participant_dir):
                session_dir = os.path.join(participant_dir, session)
                annotation_dir = os.path.join(session_dir, "annotations")
                
                if not os.path.exists(annotation_dir):
                    continue
                
                # Check if required annotation files exist
                activity_file = os.path.join(annotation_dir, "activity_annotations.json")
                actions_file = os.path.join(annotation_dir, "actions_annotations.xlsx")
                
                if not (os.path.exists(activity_file) and os.path.exists(actions_file)):
                    continue
                
                # Check if frames exist
                frames_path = os.path.join(args.frames_dir, participant, session)
                if not os.path.exists(frames_path):
                    print(f"Frames not found for {participant}/{session}, skipping")
                    continue
                
                # Parse annotations
                parsed_ann = parse_epfl_annotations(
                    participant, session, annotation_dir, frames_path, 
                    args.max_num_lines_per_gen
                )
                
                if parsed_ann is not None:
                    split_annotations.append(parsed_ann)
        
        anns_per_split[split] = split_annotations
        print(f"Loaded {len(split_annotations)} annotations for split '{split}'")
    
    return anns_per_split


if __name__ == "__main__":
    parser = HfArgumentParser((EpflPreprocessArgs, SlurmArguments))
    args, slurm_args = parser.parse_args_into_dataclasses()
    
    # Load annotations
    anns_per_split = load_epfl_dataset(args)
    
    # Convert to the format expected by run_jobs (list of dicts)
    anns_per_split_dicts = {}
    for split, annotations in anns_per_split.items():
        anns_per_split_dicts[split] = []
        for ann in annotations:
            anns_per_split_dicts[split].append({
                "participant": ann.original_ann["participant"],
                "session": ann.original_ann["session"],
                "annotation_dir": os.path.join(args.data_dir, split, ann.original_ann["participant"], ann.original_ann["session"], "annotations"),
                "frames_dir": os.path.join(args.frames_dir, ann.original_ann["participant"], ann.original_ann["session"])
            })
    
    # Submit the job
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args, anns_per_split_dicts, parse_epfl_annotation_wrapper)
    
    # Gather and save results
    import time
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    save_results(job.results(), args.splits, args.output_dir)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")