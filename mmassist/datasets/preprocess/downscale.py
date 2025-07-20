"""
Performs downscaling of Ego4D videos.

Modified from: https://github.com/facebookresearch/Ego4d/blob/main/ego4d/internal/downscale.py

Need to run this script because around 250 videos are missing in video_540ss, but present in full_scale.
"""

import datetime
import json
import math
import os
import subprocess as sp
from concurrent.futures import ThreadPoolExecutor

import glob

from tqdm.auto import tqdm
from mmassist.configs.arguments import DATA_ROOT_DIR


SRC_DIR = f"{DATA_ROOT_DIR}/datasets/ego4d_track2/v2/full_scale"
TGT_DIR = f"{DATA_ROOT_DIR}/datasets/ego4d_track2/v2/video_540ss_new"

video_ids_file = "/data/home/imzyc/project/proactive-assist/slurm_logs/missing_vids.txt"
video_ids = []
with open(video_ids_file, "r") as f:
    for line in f:
        video_ids.append(line.strip())


def call_ffmpeg(paths):
    src_path, tgt_path = paths
    assert os.path.exists(src_path)
    # https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        # TODO: try with cuda encoding for faster processing
        # '-hwaccel', 'cuda',
        # '-hwaccel_output_format', 'cuda',
        "-i",
        src_path,
        # This sweet conditional is thanks to ChatGPT :)
        "-vf",
        "scale=w=if(lt(iw\,ih)\,540\,-2):h=if(lt(iw\,ih)\,-2\,540)",  # noqa
        # '-c:a', 'copy',
        # '-c:v', 'h264_nvenc',
        # '-b:v', '5M',
        tgt_path,
        "-y",
    ]
    print(" ".join(cmd))
    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    sp.call(cmd)


def process_all(paths):
    map_fn = call_ffmpeg
    with ThreadPoolExecutor(50) as pool:
        for _ in tqdm(
            pool.map(map_fn, paths), total=len(paths), desc="Processing takes"
        ):
            continue


def main():
    map_values = []
    for video_id in video_ids:
        src_video = os.path.join(SRC_DIR, f"{video_id}.mp4")
        tgt_video = src_video.replace(SRC_DIR, TGT_DIR)
        map_values.append((src_video, tgt_video))

    process_all(map_values)


if __name__ == "__main__":
    main()
