import os
import sys
import json
import time
import argparse
import submitit
import uuid
from PIL import Image
from datasets.arrow_writer import ArrowWriter

from mmassist.datasets.utils.video_utils import img2str, resize_and_crop
from mmassist.configs.arguments import DATA_ROOT_DIR

def save_images_and_captions(
    data: list[str],
    image_root_dir: str,
    output_dir_root: str,
    img_per_file: int,
    resize_to_and_crop: int = -1,
) -> None:
    # use pyav to extract frames

    img_output_dir = os.path.join(output_dir_root, "frames")
    os.makedirs(img_output_dir, exist_ok=True)
    ann_output_dir = os.path.join(output_dir_root, "annotations")
    os.makedirs(ann_output_dir, exist_ok=True)

    writer = None
    for ori_idx, d in enumerate(data):
        if ori_idx % img_per_file == 0:
            file_name = str(uuid.uuid4())
            arrow_file = os.path.join(img_output_dir, f"{file_name}.arrow")
            writer = ArrowWriter(path=arrow_file)
            captions = {"prompt": d["conversations"][0]["value"], "captions": []}
        captions["captions"].append(d["conversations"][1]["value"])

        img_file = os.path.join(image_root_dir, d["image"])
        img = Image.open(img_file)
        if resize_to_and_crop > 0:
            # img = resize_and_crop(img, resize_to_and_crop)
            # resize the images as suggested in https://arxiv.org/pdf/2402.07865
            img = img.resize((resize_to_and_crop, resize_to_and_crop))
        writer.write({"frame": img2str(img)})

        if ori_idx % img_per_file == img_per_file - 1 or ori_idx == len(data) - 1:
            writer.finalize()
            print(f"Saved {ori_idx + 1 % img_per_file} images to {arrow_file}")
            ann_file = os.path.join(ann_output_dir, f"{file_name}.json")
            with open(ann_file, "w") as f:
                json.dump(captions, f, indent=2)


class Task:
    def __call__(self, args):

        # global rank
        job_env = submitit.JobEnvironment()
        local_rank = job_env.local_rank
        global_rank = job_env.global_rank
        num_tasks = job_env.num_tasks

        # load annotation file
        with open(args.ann_file, "r") as f:
            data = json.load(f)

        partition_data = data[global_rank::num_tasks]
        print(f"Rank {global_rank}/{num_tasks} processing {len(partition_data)} images")

        start_time = time.time()
        save_images_and_captions(
            partition_data,
            args.image_root,
            args.output_dir,
            args.images_per_file,
            args.resize_to,
        )
        print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=1,
        tasks_per_node=140,
        cpus_per_task=1,
        slurm_partition="q1",
        account="ar-ai-research-interns",
        name="save_images",
        mem_gb=1024,
        timeout_min=60 * 24,
    )
    job = executor.submit(Task(), args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ann_file",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        help="Path to the LLAVA annotation file",
    )
    args.add_argument(
        "--image_root",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/LLaVA-Pretrain/images",
        help="Path to the image files",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/llava",
        help="Path to the output directory",
    )
    args.add_argument(
        "--images_per_file",
        type=int,
        default=2000,
        help="Number of images per arrow file",
    )
    args.add_argument(
        "--resize_to",
        type=int,
        default=384,
        help="Center crop the frames to a square and resize to this size",
    )
    args = args.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    sys.exit(main(args))
