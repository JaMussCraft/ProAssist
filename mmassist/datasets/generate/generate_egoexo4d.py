import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import HfArgumentParser
from tqdm import tqdm

from mmassist.datasets.generate.dialog_simulation import ParsedVideoAnns, generate_from_annotation
from mmassist.datasets.generate.auto_eval import auto_eval_generated_conversations
from mmassist.configs.arguments import DATA_ROOT_DIR
from mmassist.datasets.generate.openrouter_utils import LLMGenerator
from mmassist.datasets.generate.frame_utils import (
    load_frames_from_arrow,
    get_frame_at_timestamp,
    describe_frame_with_llm,
)
from mmassist.datasets.generate.dialog_simulation import FRAME_DESCRIPTION_PROMPT


@dataclass
class EgoExo4DPreprocessArgs:
    data_dir: str = f"{DATA_ROOT_DIR}/datasets/egoexo4d/annotations"
    frames_dir: str = f"{DATA_ROOT_DIR}/processed_data/egoexo4d/frames"
    splits: str = "train,val"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/egoexo4d/generated_dialogs"
    llm: str = "google/gemini-2.5-flash"
    user_types: str = "no_talk@2,talk_some@4,talk_more@4"
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = True
    max_num_lines_per_gen: int = 50
    # Frame incorporation options
    use_frames: str = "none"  # Options: "none", "video", "frames", "descriptions"
    frames_fps: float = 2.0  # FPS of the extracted frames in Arrow files
    video_llm: str = "google/gemini-2.5-pro"  # LLM for video-based generation (Option 1)
    frame_desc_llm: str = "google/gemini-2.5-flash"  # LLM for frame descriptions (Option 3)


def load_atomic_descriptions(data_dir: str, split: str) -> Dict:
    """
    Load atomic description annotations for a split.
    
    Args:
        data_dir: Path to the annotations directory
        split: Split name (train or val)
        
    Returns:
        Dictionary with atomic description data
    """
    atomic_file = os.path.join(data_dir, f"atomic_descriptions_{split}_filtered_longest.json")
    with open(atomic_file, 'r') as f:
        return json.load(f)


def load_keystep_annotations(data_dir: str, split: str) -> Dict:
    """
    Load keystep annotations for a split.
    
    Args:
        data_dir: Path to the annotations directory
        split: Split name (train or val)
        
    Returns:
        Dictionary with keystep data
    """
    keystep_file = os.path.join(data_dir, f"keystep_{split}.json")
    with open(keystep_file, 'r') as f:
        return json.load(f)


def load_take_uid_to_task_name_mapping(data_dir: str) -> Dict[str, str]:
    """
    Load takes.json and create a mapping from take_uid to task_name.
    
    Args:
        data_dir: Path to the datasets/egoexo4d directory
        
    Returns:
        Dictionary mapping take_uid to take_name
    """
    # Adjust path to go up from annotations to the parent directory
    takes_file = os.path.join(os.path.dirname(data_dir), "takes.json")
    
    with open(takes_file, 'r') as f:
        takes_data = json.load(f)
    
    # Create mapping from take_uid to take_name
    take_uid_to_name = {}
    for take in takes_data:
        take_uid = take.get('take_uid')
        task_name = take.get('task_name')
        if take_uid and task_name:
            take_uid_to_name[take_uid] = task_name
    
    return take_uid_to_name


def parse_egoexo4d_annotations(
    split: str,
    take_uid: str,
    atomic_data: Dict,
    keystep_data: Dict,
    take_uid_to_task_name, 
    max_num_lines_per_gen: int = 50,
    frames_dir: Optional[str] = None,
    use_frames: str = "none",
    frames_fps: float = 2.0,
    frame_desc_llm: Optional[LLMGenerator] = None,
) -> Optional[ParsedVideoAnns]:
    """
    Parse EgoExo4D annotations into the format expected by the dialog generation pipeline.
    
    Args:
        split: Split name (train or val)
        take_uid: Take UID
        atomic_data: Atomic description data
        keystep_data: Keystep annotation data
        max_num_lines_per_gen: Maximum number of lines per generation clip
        frames_dir: Directory containing frame Arrow files (for frame incorporation)
        use_frames: Frame incorporation mode ("none", "video", "frames", "descriptions")
        frames_fps: FPS of extracted frames
        frame_desc_llm: LLM for generating frame descriptions (Option 3)
        
    Returns:
        ParsedVideoAnns object or None if parsing fails
    """
    try:
        # Get atomic descriptions for this take_uid
        if take_uid not in atomic_data['annotations']:
            print(f"No atomic annotations found for {take_uid}")
            return None
        
        take_annotation = atomic_data['annotations'][take_uid][0]
        
        # Filter out annotation's descriptions with unsure=True
        atomic_annotations = []
        for ann in take_annotation["descriptions"]:
            # break
            if not ann["unsure"]:
                atomic_annotations.append(ann)
        
        if not atomic_annotations:
            print(f"No valid atomic annotations (all were unsure) for {take_uid}")
            return None
        
        # Load frames if needed for description generation (Option 3)
        frames_data = None
        if use_frames == "descriptions" and frames_dir is not None:
            arrow_file = os.path.join(frames_dir, f"{take_uid}.arrow")
            if os.path.exists(arrow_file):
                try:
                    frames_data = load_frames_from_arrow(arrow_file)
                    print(f"         Loaded {len(frames_data)} frames from {arrow_file}")
                except Exception as e:
                    print(f"         Warning: Failed to load frames from {arrow_file}: {e}")
                    frames_data = None
            else:
                print(f"         Warning: Frame file not found: {arrow_file}")
        
        # Create step descriptions from atomic annotations
        all_descriptions = []
        for ann in atomic_annotations:
            timestamp = ann['timestamp']
            text = ann['text']
            
            # Generate frame description if using Option 3
            frame_desc = ""
            if use_frames == "descriptions" and frames_data is not None and frame_desc_llm is not None:
                try:
                    frame = get_frame_at_timestamp(frames_data, timestamp, frames_fps)
                    if frame is not None:
                        frame_desc = describe_frame_with_llm(frame, frame_desc_llm, FRAME_DESCRIPTION_PROMPT)
                        frame_desc = f" (image shows: {frame_desc})"
                except Exception as e:
                    print(f"         Warning: Failed to describe frame at {timestamp}s: {e}")
            
            # Create time-stamped description with optional frame description
            step_desc = f"[{timestamp:.1f}s] {text}{frame_desc}"
            
            all_descriptions.append({
                "timestamp": timestamp,
                "text": text,
                "step": step_desc
            })
        
        # Create all step descriptions string
        all_step_descriptions = "\n".join([desc["step"] for desc in all_descriptions])
        
        # Split into clips based on max_num_lines_per_gen
        clips = []
        num_lines_in_clip = 0
        clip_start_idx = 0
        clip_start_time = -1
        clip_end_time = -1
        
        for idx, desc in enumerate(all_descriptions):
            num_lines_in_clip += 1
            
            if clip_start_time < 0:
                clip_start_time = desc["timestamp"]
            clip_end_time = desc["timestamp"]
            
            if (num_lines_in_clip >= max_num_lines_per_gen or 
                idx == len(all_descriptions) - 1):
                
                # Create clip description
                clip_description = "\n".join([
                    all_descriptions[s_idx]["step"] 
                    for s_idx in range(clip_start_idx, idx + 1)
                ])
                
                clips.append((clip_start_time, clip_end_time, clip_description))
                clip_start_idx = idx + 1
                clip_start_time = -1
                clip_end_time = -1
                num_lines_in_clip = 0
        
        # Calculate metrics
        total_duration = atomic_annotations[-1]['timestamp']
        
        # For atomic annotations, we assume they cover the full duration
        # since each annotation is a point in time
        ann_ratio = 1.0
        
        # Get keystep annotations if available
        keystep_anns = None
        if take_uid in keystep_data.get('annotations', {}):
            keystep_anns = keystep_data['annotations'][take_uid]
        
        original_ann = {
            "take_uid": take_uid,
            "atomic_annotations": atomic_annotations,
        }
        
        # Add keystep annotations to original_ann if available
        if keystep_anns:
            original_ann["keystep_annotations"] = keystep_anns
        
        parsed_ann = ParsedVideoAnns(
            dataset="egoexo4d",
            domain="cooking",
            knowledge_type="cooking recipe",
            video_uid=take_uid,
            goal_description=take_uid_to_task_name[take_uid],
            all_step_descriptions=all_step_descriptions,
            clips=clips,
            duration=total_duration,
            ann_ratio=ann_ratio,
            num_steps=len(atomic_annotations),
            num_substeps=0,
            original_ann=original_ann
        )
        
        return parsed_ann
        
    except Exception as e:
        print(f"Error parsing annotations for {take_uid}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_egoexo4d_dataset(args: EgoExo4DPreprocessArgs) -> Dict[str, List[ParsedVideoAnns]]:
    """
    Load all EgoExo4D annotations for the specified splits.
    
    Args:
        args: Preprocessing arguments
        
    Returns:
        Dictionary mapping split names to lists of ParsedVideoAnns
    """
    anns_per_split = {}
    
    # Initialize frame description LLM if needed (Option 3)
    frame_desc_llm = None
    if args.use_frames == "descriptions":
        print(f"Initializing frame description LLM: {args.frame_desc_llm}")
        frame_desc_llm = LLMGenerator.build(model_id=args.frame_desc_llm)
    
    for split in args.splits.split(","):
        print(f"\nLoading {split} split...")
        
        # Load atomic descriptions and keystep annotations
        try:
            atomic_data = load_atomic_descriptions(args.data_dir, split)
            keystep_data = load_keystep_annotations(args.data_dir, split)
        except FileNotFoundError as e:
            print(f"Annotation files not found for split {split}: {e}")
            continue

        take_uid_to_task_name = load_take_uid_to_task_name_mapping(args.data_dir)
        
        split_annotations = []
        
        # Get all take_uids from atomic annotations
        take_uids = list(atomic_data.get('annotations', {}).keys())
        print(f"Found {len(take_uids)} take_uids in atomic annotations")
        
        # Parse each take_uid
        for take_uid in tqdm(take_uids, desc=f"Parsing {split}"):
            # Parse annotations
            parsed_ann = parse_egoexo4d_annotations(
                split, take_uid, atomic_data, keystep_data, take_uid_to_task_name,
                args.max_num_lines_per_gen,
                frames_dir=args.frames_dir if args.use_frames != "none" else None,
                use_frames=args.use_frames,
                frames_fps=args.frames_fps,
                frame_desc_llm=frame_desc_llm,
            )
            
            if parsed_ann is not None:
                split_annotations.append(parsed_ann)
        
        anns_per_split[split] = split_annotations
        print(f"Loaded {len(split_annotations)} annotations for split '{split}'")
    
    return anns_per_split


def run_local_jobs(
    args: EgoExo4DPreprocessArgs,
    anns_per_split: Dict[str, List[ParsedVideoAnns]],
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
            vid = ann.video_uid
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
        for ann in all_anns_to_run[:5]:  # Show first 5
            vid = ann.video_uid
            print(f"  {vid}")
        if len(all_anns_to_run) > 5:
            print(f"  ... and {len(all_anns_to_run) - 5} more")
    
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
            
            for parsed_ann in tqdm(anns_to_run_per_split[split], desc=f"Processing {split}"):
                if llm is None:
                    # build llm
                    # For video mode (Option 1), use a more capable model
                    model_to_use = args.video_llm if args.use_frames == "video" else args.llm
                    print(f"Initializing LLM: {model_to_use}")
                    llm = LLMGenerator.build(model_id=model_to_use)
                    gen_args = {
                        "llm": model_to_use,
                        "user_types": args.user_types,
                        "num_repeats": args.num_repeats,
                        "sampling_params": llm.default_sampling_args,
                        "use_frames": args.use_frames,
                        "frames_fps": args.frames_fps,
                    }
                
                # generate the outputs (parsed_ann is already a ParsedVideoAnns object)
                outputs = generate_from_annotation(
                    parsed_ann,
                    llm,
                    user_types=user_types,
                    num_repeats=args.num_repeats,
                    min_ann_ratio=args.min_ann_ratio,
                    filter_by_llm=args.filter_by_llm,
                    frames_dir=args.frames_dir if args.use_frames != "none" else None,
                    use_frames=args.use_frames,
                    frames_fps=args.frames_fps,
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
        
        # also load the samples that have been processed before
        for ann in anns_to_load_per_split[split]:
            # load the outputs
            vid = ann.video_uid
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
    parser = HfArgumentParser(EgoExo4DPreprocessArgs)
    args = parser.parse_args_into_dataclasses()[0]
    
    print("EgoExo4D Dialog Generation - Local Mode")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LLM model: {args.llm}")
    print(f"Splits: {args.splits}")
    print(f"User types: {args.user_types}")
    print(f"Force rerun: {args.force_rerun}")
    print(f"Max lines per gen: {args.max_num_lines_per_gen}")
    print(f"Frame mode: {args.use_frames}")
    if args.use_frames != "none":
        print(f"  Frames FPS: {args.frames_fps}")
        if args.use_frames == "video":
            print(f"  Video LLM: {args.video_llm}")
        elif args.use_frames == "descriptions":
            print(f"  Frame description LLM: {args.frame_desc_llm}")
    print("=" * 50)
    
    # Load annotations
    print("Loading EgoExo4D annotations...")
    anns_per_split = load_egoexo4d_dataset(args)

    anns_per_split["train"] = anns_per_split["train"][:1] # temp
    anns_per_split["val"] = anns_per_split["val"][:1] # temp
    
    if not any(anns_per_split.values()):
        print("No annotations found! Please check your data directory.")
        exit(1)
    
    # Process locally
    print("Starting local processing...")
    start_time = time.time()
    
    split_outputs = run_local_jobs(args, anns_per_split)
    
    # Save results with auto-evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    save_local_results(split_outputs, args.splits, args.output_dir)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nProcessing completed in {total_time:.2f} minutes")
    print("=" * 50)
