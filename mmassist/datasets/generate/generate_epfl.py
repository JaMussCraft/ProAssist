import os
import json
import pandas as pd
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import HfArgumentParser
from tqdm import tqdm

from mmassist.datasets.generate.dialog_simulation import ParsedVideoAnns, generate_from_annotation
from mmassist.datasets.generate.auto_eval import auto_eval_generated_conversations
from mmassist.configs.arguments import DATA_ROOT_DIR
from mmassist.datasets.generate.openrouter_utils import LLMGenerator


@dataclass
class EpflPreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/epfl/annotations"
    frames_dir: str = f"{DATA_ROOT_DIR}/processed_data/epfl/frames"
    splits: str = "train,test"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/epfl/generated_dialogs"
    llm: str = "google/gemini-2.5-flash"
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
                frames_path = f"/projects/beto/proassist_data/processed_data/epfl/frames/{split}_{participant}_{session}_hololens_compressed.arrow"
                if not os.path.exists(frames_path):
                    print(f"Frames not found for {participant}/{session}, skipping")
                    continue
                
                # Parse annotations
                parsed_ann = parse_epfl_annotations(
                    split, participant, session, annotation_dir, 
                    args.max_num_lines_per_gen
                )
                
                if parsed_ann is not None:
                    split_annotations.append(parsed_ann)
        
        anns_per_split[split] = split_annotations
        print(f"Loaded {len(split_annotations)} annotations for split '{split}'")
    
    return anns_per_split


def run_local_jobs(
    args: EpflPreprocessArgs,
    anns_per_split: Dict[str, List[dict]],
    ann_parse_func: callable,
):
    """
    Local version of the run_jobs function that processes annotations sequentially
    without SLURM parallelization.
    """
    print("Starting local job execution...")
    
    splits = args.splits.split(",")
    
    # get the samples to run/load for each split
    anns_to_run_per_split = {}
    anns_to_load_per_split = {}
    
    for split in splits:
        # get the anns in the split
        all_anns_in_split = anns_per_split[split]
        
        # filter out the anns that already been processed
        all_anns_to_run, all_anns_to_load = [], []
        for ann in all_anns_in_split:
            vid = ann.get("video_uid") or f"{split}_{ann['participant']}_{ann['session']}"
            output_file = os.path.join(args.output_dir, split, f"{vid}.json")
            if args.force_rerun:
                all_anns_to_run.append(ann)
            else:
                try:
                    with open(output_file, "r") as f:
                        json.load(f)  # Check if file exists and is valid JSON
                    all_anns_to_load.append(ann)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_anns_to_run.append(ann)
        
        anns_to_run_per_split[split] = all_anns_to_run
        anns_to_load_per_split[split] = all_anns_to_load
        
        print(f"{split}: {len(all_anns_to_run)} files to process, {len(all_anns_to_load)} already processed")
        for ann in all_anns_to_run:
            vid = ann.get("video_uid") or f"{split}_{ann['participant']}_{ann['session']}"
            print(f"  {vid}")
    
    # parse user types
    user_types = []
    for user_type_with_rep in args.user_types.split(","):
        user_type, num_repeats = user_type_with_rep.split("@")
        user_types.extend([user_type] * int(num_repeats))
    
    llm, gen_args = None, None
    split_outputs = {}
    
    for split in splits:
        if split not in split_outputs:
            split_outputs[split] = []
        
        # process the samples
        if anns_to_run_per_split[split]:
            print(f"\nProcessing {len(anns_to_run_per_split[split])} annotations for split '{split}'...")
            
            for ann in tqdm(anns_to_run_per_split[split], desc=f"Processing {split}"):
                if llm is None:
                    # build llm
                    print(f"Initializing LLM: {args.llm}")
                    llm = LLMGenerator.build(model_id=args.llm)
                    gen_args = {
                        "llm": args.llm,
                        "user_types": args.user_types,
                        "num_repeats": args.num_repeats,
                        "sampling_params": llm.default_sampling_args,
                    }
                
                # parse the annotation
                parsed_ann: ParsedVideoAnns | None = ann_parse_func(ann, args.max_num_lines_per_gen)
                if parsed_ann is None:
                    continue
                
                # generate the outputs
                outputs = generate_from_annotation(
                    parsed_ann,
                    llm,
                    user_types=user_types,
                    num_repeats=args.num_repeats,
                    min_ann_ratio=args.min_ann_ratio,
                    filter_by_llm=args.filter_by_llm,
                )
                
                # save the output
                output_dir = os.path.join(args.output_dir, split)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{parsed_ann.video_uid}.json")
                
                if isinstance(outputs, str):
                    with open(output_file, "w") as f:
                        json.dump({"reason_to_exclude": outputs}, f, indent=2)
                else:
                    outputs_dict = outputs.to_dict()
                    outputs_dict["gen_args"] = gen_args
                    with open(output_file, "w") as f:
                        json.dump(outputs_dict, f, indent=2)
                    
                    if "reason_to_exclude" not in outputs_dict:
                        split_outputs[split].append(outputs_dict)

                break # for now
        
        # also load the samples that have been processed before
        for ann in anns_to_load_per_split[split]:
            # load the outputs
            vid = ann.get("video_uid") or f"{split}_{ann['participant']}_{ann['session']}"
            output_file = os.path.join(args.output_dir, split, f"{vid}.json")
            try:
                with open(output_file, "r") as f:
                    outputs = json.load(f)
                if "reason_to_exclude" not in outputs:
                    split_outputs[split].append(outputs)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not load {output_file}")
    
    return split_outputs


def save_local_results(split_outputs: Dict[str, List[dict]], splits: str, output_dir: str):
    """Save results from local execution and apply auto-evaluation"""
    for split in splits.split(","):
        split_data = split_outputs.get(split, [])
        
        # auto-evaluate all the generated dialogs
        print(f"Auto-evaluating {len(split_data)} dialogs for split '{split}'...")
        for idx, output in enumerate(split_data):
            try:
                split_data[idx] = auto_eval_generated_conversations(output)
            except Exception as e:
                print(f"Warning: Failed to auto-evaluate dialog {idx}: {e}")
        
        # save the outputs
        print(f"Saving '{split}' split of {len(split_data)} videos")
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = HfArgumentParser(EpflPreprocessArgs)
    args = parser.parse_args_into_dataclasses()[0]
    
    print("EPFL Dialog Generation - Local Mode")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LLM model: {args.llm}")
    print(f"Splits: {args.splits}")
    print(f"User types: {args.user_types}")
    print(f"Force rerun: {args.force_rerun}")
    print("=" * 50)
    
    # Load annotations
    print("Loading EPFL annotations...")
    anns_per_split = load_epfl_dataset(args)
    
    if not any(anns_per_split.values()):
        print("No annotations found! Please check your data directory.")
        exit(1)
    
    # Convert to the format expected by processing functions
    anns_per_split_dicts = {}
    for split, annotations in anns_per_split.items():
        anns_per_split_dicts[split] = []
        for ann in annotations:
            anns_per_split_dicts[split].append({
                "split": split,
                "participant": ann.original_ann["participant"],
                "session": ann.original_ann["session"],
                "annotation_dir": os.path.join(args.data_dir, split, ann.original_ann["participant"], ann.original_ann["session"], "annotations"),
                "video_uid": ann.video_uid
            })
    
    # Process locally
    print("Starting local processing...")
    start_time = time.time()
    
    split_outputs = run_local_jobs(args, anns_per_split_dicts, parse_epfl_annotation_wrapper)
    
    # Save results with auto-evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    save_local_results(split_outputs, args.splits, args.output_dir)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nProcessing completed in {total_time:.2f} minutes")
    print("=" * 50)