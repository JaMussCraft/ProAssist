import os
import av
import sys
import json
import time
import argparse
import submitit
from datasets.arrow_writer import ArrowWriter

from mmassist.datasets.utils.video_utils import img2str, resize_and_crop
from mmassist.configs.arguments import DATA_ROOT_DIR

def save_videos_and_anns(args, data_groups: list[dict]) -> None:
    # sanity check for video files
    for group in data_groups:
        for d in group["anns"]:
            video_file = os.path.join(args.video_dir, f"{d['id']}.webm")
            try:
                container = av.open(video_file)
                container.close()
            except:
                raise ValueError(f"Failed to open {video_file}")

    # make output directories
    frame_output_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_output_dir, exist_ok=True)
    ann_output_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(ann_output_dir, exist_ok=True)

    # extract frames from videos and save each group into an .arrow file
    for group in data_groups:
        output_file = os.path.join(frame_output_dir, f"{group['uid']}.arrow")
        writer = ArrowWriter(path=output_file)
        frame_idx = 0
        for d in group["anns"]:
            video_file = os.path.join(args.video_dir, f"{d['id']}.webm")
            # use pyav to extract frames
            container = av.open(video_file)
            stream = container.streams.video[0]  # take first video stream
            original_fps = int(stream.average_rate)
            step_size = int((original_fps + 0.5) // args.target_fps)

            start_idx = frame_idx
            # process the frames and save into an .arrow file
            for ori_idx, frame in enumerate(container.decode(stream)):
                if ori_idx % step_size == 0:
                    img = frame.to_image()
                    if args.center_crop_and_resize_to > 0:
                        img = resize_and_crop(img, args.center_crop_and_resize_to)
                    writer.write({"frame": img2str(img)})
                    frame_idx += 1
            d["frame_start_idx"] = start_idx
            d["frame_end_idx"] = frame_idx
            container.close()
        writer.finalize()

        # save ann
        group["num_frames"] = frame_idx
        group["fps"] = args.target_fps
        ann_file = os.path.join(ann_output_dir, f"{group['uid']}.json")
        with open(ann_file, "w") as f:
            json.dump(group, f, indent=2)


def run_jobs(args):
    # global rank
    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    global_rank = job_env.global_rank
    num_tasks = job_env.num_tasks

    all_video_groups = []
    train_anns = json.load(open(os.path.join(args.label_dir, "train.json")))
    # split train_anns into groups of size args.videos_per_file
    for i in range(0, len(train_anns), args.videos_per_file):
        anns = train_anns[i : i + args.videos_per_file]
        all_video_groups.append({"anns": anns, "uid": anns[0]["id"], "split": "train"})
    valid_anns = json.load(open(os.path.join(args.label_dir, "validation.json")))
    for i in range(0, len(valid_anns), args.videos_per_file):
        anns = valid_anns[i : i + args.videos_per_file]
        all_video_groups.append({"anns": anns, "uid": anns[0]["id"], "split": "valid"})

    partition_data = all_video_groups[global_rank::num_tasks]
    print(f"Rank {global_rank}/{num_tasks} processing {len(partition_data)} videos")

    start_time = time.time()
    save_videos_and_anns(args, partition_data)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=8,
        tasks_per_node=192,
        cpus_per_task=1,
        slurm_partition="q1",
        # slurm_account="ar-ai-research-interns",
        name="save_images",
        mem_gb=1024,
        timeout_min=60 * 24,
    )
    job = executor.submit(run_jobs, args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--label_dir", type=str, default=f"{DATA_ROOT_DIR}/datasets/sthsth-v2/labels"
    )
    args.add_argument(
        "--video_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/sthsth-v2/videos/videos",
    )
    args.add_argument(
        "--output_dir", type=str, default=f"{DATA_ROOT_DIR}/processed_data/sthsthv2"
    )
    args.add_argument("--videos_per_file", type=int, default=256)
    args.add_argument("--target_fps", type=int, default=2)
    args.add_argument("--center_crop_and_resize_to", type=int, default=384)
    args = args.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    sys.exit(main(args))
