import json
import os
import argparse
import multiprocessing as mp
import time

from mmassist.datasets.utils.ego4d_utils import process_ego4d_narration_data
from mmassist.configs.arguments import DATA_ROOT_DIR

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ego4d_data_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/ego4d_track2/",
        help="Path to the ego4d dataset directory",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/ego4d",
        help="Path to the output directory",
    )
    args.add_argument(
        "--target_fps", type=int, default=2, help="Target FPS for the output frames"
    )
    args.add_argument(
        "--num_proc",
        type=int,
        default=mp.cpu_count(),
        help="Number of processes for parallel processing",
    )
    args = args.parse_args()
    args.ego4d_data_version = version = "v2"

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    print("Loading metadata of all videos...")
    all_video_meta_file = os.path.join(args.ego4d_data_dir, "ego4d.json")
    with open(all_video_meta_file, "r") as f:
        all_video_meta = json.load(f)
    all_video_meta = {v["video_uid"]: v for v in all_video_meta["videos"]}

    print("Loading narration annotations of all videos...")
    all_narr_ann_file = os.path.join(
        args.ego4d_data_dir, version, "annotations", f"narration.json"
    )
    with open(all_narr_ann_file, "r") as f:
        all_narr_anns = json.load(f)

    all_video_uids = list(all_narr_anns.keys())

    # multiprocessing yse process_ego4d_narration_data
    print("Processing narration annotations...")
    start_time = time.time()

    inputs = []
    for video_id in all_video_uids:
        inputs.append((video_id, all_narr_anns[video_id], all_video_meta[video_id]))

    def process_video(inputs):
        video_id, video_narr_anns, meta = inputs

        # # skip videos that has redacted annotations
        # # Note: this will skip 678 'redacted' and 383 with 'redacted_partial' videos
        # if video_narr_anns["status"] != "complete":
        #     return

        for narr_annotator_id in range(1, 5):
            if f"narration_pass_{narr_annotator_id}" not in video_narr_anns:
                continue
            if not video_narr_anns[f"narration_pass_{narr_annotator_id}"]["narrations"]:
                continue

            try:
                process_ego4d_narration_data(
                    video_id=video_id,
                    video_narr_anns=video_narr_anns,
                    narr_annotator_id=narr_annotator_id,
                    metadata=meta,
                    target_fps=args.target_fps,
                    output_dir=args.output_dir,
                )
            except:
                # traceback
                from traceback import print_exc

                print_exc()
                print(video_id, narr_annotator_id, "failed")
                continue

    with mp.Pool(args.num_proc) as pool:
        pool.map(process_video, inputs)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time: {elapsed_time:.2f} minutes")
