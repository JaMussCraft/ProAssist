import os
import sys
import time
import glob
import argparse
import submitit

from tqdm import tqdm
from mmassist.datasets.utils.video_utils import extract_frames_to_arrow
from mmassist.configs.arguments import DATA_ROOT_DIR


def get_ego4d_video_files(args) -> tuple[list[list[str]], list[str]]:
    fmt = ".mp4"

    # As Ego4D goal-step videos might be a group of videos concatenated together,
    # we save all the frames from a grouped video into a single arrow file, named
    # after the first video in the group.
    import ast
    import pandas as pd

    vf_to_vfs = {}
    video_groups = pd.read_csv(args.video_group_file, delimiter="\t")
    for entry in video_groups.to_dict("records"):
        vg = ast.literal_eval(entry["video_group"])
        vf_to_vfs[f"{vg[0]}{fmt}"] = [f"{v}{fmt}" for v in vg]

    video_file_names = [f for f in os.listdir(args.video_dir) if f.endswith(fmt)]

    video_files, video_ids = [], []
    for fn in video_file_names:
        file_names_to_concat = vf_to_vfs.get(fn, [fn])
        abs_file_paths = []
        for file_name in file_names_to_concat:
            f = os.path.join(args.video_dir, file_name)
            assert os.path.exists(f), f"Video {f} does not exist"
            abs_file_paths.append(f)
        if len(abs_file_paths) == 1:
            continue
        video_files.append(abs_file_paths)
        video_ids.append(file_names_to_concat[0].replace(fmt, ""))
    output_files = [os.path.join(args.output_dir, f"{f}.arrow") for f in video_ids]
    return video_files, output_files


def get_holoassist_video_files(args) -> tuple[list[list[str]], list[str]]:
    video_ids = [d for d in os.listdir(args.video_dir)]
    video_files = []
    for vid in video_ids:
        f = os.path.join(args.video_dir, vid, "Export_py/Video_pitchshift.mp4")
        assert os.path.exists(f), f"Video {f} does not exist"
        video_files.append([f])
    output_files = [os.path.join(args.output_dir, f"{f}.arrow") for f in video_ids]
    return video_files, output_files


def get_epickitchens_video_files(args) -> tuple[list[list[str]], list[str]]:
    ek55_video_dir = os.path.join(args.video_dir, "EK55/videos")
    ek55_video_files = glob.glob(os.path.join(ek55_video_dir, "*/*/*.MP4"))
    ek55_video_ids = [f.split("/")[-1].split(".")[0] for f in ek55_video_files]

    ek100_video_dir = os.path.join(args.video_dir, "EK100")
    ek100_video_files = glob.glob(os.path.join(ek100_video_dir, "*/videos/*.MP4"))
    ek100_video_ids = [f.split("/")[-1].split(".")[0] for f in ek100_video_files]

    video_ids = ek55_video_ids + ek100_video_ids
    video_files = []
    for f in ek55_video_files + ek100_video_files:
        assert os.path.exists(f), f"Video {f} does not exist"
        video_files.append([f])
    output_files = [os.path.join(args.output_dir, f"{i}.arrow") for i in video_ids]
    return video_files, output_files


def get_egoexolearn_video_files(args) -> tuple[list[list[str]], list[str]]:
    fmt = ".mp4"
    video_file_names = [f for f in os.listdir(args.video_dir) if f.endswith(fmt)]
    video_ids = [f.replace(fmt, "") for f in video_file_names]
    video_files = []
    for fn in video_file_names:
        f = os.path.join(args.video_dir, fn)
        assert os.path.exists(f), f"Video {f} does not exist"
        video_files.append([f])
    output_files = [os.path.join(args.output_dir, f"{i}.arrow") for i in video_ids]
    return video_files, output_files


def get_wtag_video_files(args) -> tuple[list[list[str]], list[str]]:
    video_ids = [d for d in os.listdir(args.video_dir)]
    video_files = []
    for vid in video_ids:
        f = os.path.join(args.video_dir, vid, "Video/Video.mpeg")
        assert os.path.exists(f), f"Video {f} does not exist"
        video_files.append([f])
    output_files = [os.path.join(args.output_dir, f"{f}.arrow") for f in video_ids]
    return video_files, output_files



def get_assembly101_video_files(args) -> tuple[list[list[str]], list[str]]:
    seq_ids = [d for d in os.listdir(args.video_dir)]
    selected_cam_ids = [
        "HMC_21110305_mono10bit",
        "HMC_21179183_mono10bit",
        "HMC_84355350_mono10bit",
        "HMC_84358933_mono10bit",
    ]
    video_ids = []
    video_files = []
    for seq_id in seq_ids:
        for cam_id in selected_cam_ids:
            video_file = os.path.join(args.video_dir, seq_id, f"{cam_id}.mp4")
            if os.path.exists(video_file):
                video_files.append([video_file])
                video_ids.append(f"{seq_id}__{cam_id}")
    output_files = [os.path.join(args.output_dir, f"{f}.arrow") for f in video_ids]
    return video_files, output_files


def get_custom_video_files(args) -> tuple[list[list[str]], list[str]]:
    video_files = [
        os.path.join(args.video_dir, "P29/videos", "P29_01.MP4"),
        os.path.join(args.video_dir, "P29/videos", "P29_05.MP4"),
        os.path.join(args.video_dir, "P30/videos", "P30_05.MP4"),
        os.path.join(args.video_dir, "P30/videos", "P30_08.MP4"),
    ]
    video_ids = [f.split("/")[-1].split(".")[0] for f in video_files]
    video_files = [[f] for f in video_files]
    output_files = [os.path.join(args.output_dir, f"{f}.arrow") for f in video_ids]
    return video_files, output_files


dataset_to_get_func = {
    "ego4d": get_ego4d_video_files,
    "holoassist": get_holoassist_video_files,
    "epickitchens": get_epickitchens_video_files,
    "egoexolearn": get_egoexolearn_video_files,
    "wtag": get_wtag_video_files,
    "assembly101": get_assembly101_video_files,
    "custom": get_custom_video_files,
}


def run_jobs(args):

    # global rank
    job_env = submitit.JobEnvironment()
    global_rank = job_env.global_rank
    num_tasks = job_env.num_tasks

    video_files, output_files = dataset_to_get_func[args.dataset](args)
    video_files = video_files[global_rank::num_tasks]
    output_files = output_files[global_rank::num_tasks]

    start_time = time.time()
    for video_files, out_file in tqdm(zip(video_files, output_files)):
        print("Processing videos: ", video_files)
        rotate = 90 if args.dataset == "assembly101" else -1
        resize_to = args.center_crop_and_resize_to
        try:
            extract_frames_to_arrow(
                video_files, out_file, args.target_fps, resize_to, rotate
            )
        except Exception as e:
            print(f"Failed to process {video_files} with error: {e}")
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=args.num_nodes,
        tasks_per_node=36,
        cpus_per_task=1,
        account="engin1",
        slurm_partition="standard",
        name=f"ef_{args.dataset}",
        mem_gb=128,
        timeout_min=60 * 24,
    )
    job = executor.submit(run_jobs, args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "ego4d",
            "holoassist",
            "epickitchens",
            "egoexolearn",
            "wtag",
            "assembly101",
            "custom",
        ],
        help="The video dataset to process",
    )
    args.add_argument(
        "--video_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/ego4d_track2/v2/video_540ss",
        help="Path to the video files",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/ego4d/frames",
        help="Path to the output directory",
    )
    args.add_argument(
        "--video_group_file",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/ego4d_track2/v2/annotations/goalstep_video_groups.tsv",  # noqa
        help="The video group file for Ego4D goalstep.",
    )
    args.add_argument(
        "--target_fps", type=int, default=2, help="Target FPS for the output frames"
    )
    args.add_argument(
        "--center_crop_and_resize_to",
        type=int,
        default=384,
        help="Center crop the frames to a square and resize to this size",
    )
    args.add_argument("--num_nodes", type=int, default=1)
    args = args.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    vfs, ofs = dataset_to_get_func[args.dataset](args)
    print("Total videos: ", len(vfs))
    sys.exit(main(args))
