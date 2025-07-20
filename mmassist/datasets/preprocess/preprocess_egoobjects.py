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

def save_images_and_anns(args, data_groups: list[list[dict]]) -> None:
    # sanity check for video files
    for group in data_groups:
        for d in group:
            img_file = os.path.join(args.image_root, d["url"])
            assert os.path.exists(img_file), f"Image {img_file} does not exist"

    # make output directories
    frame_output_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_output_dir, exist_ok=True)
    ann_output_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(ann_output_dir, exist_ok=True)

    # save group of images into an .arrow file
    for group in data_groups:
        uid = group[0]["id"]

        output_file = os.path.join(frame_output_dir, f"{uid}.arrow")
        writer = ArrowWriter(path=output_file)
        for d in group:
            img_file = os.path.join(args.image_root, d["url"])
            img = Image.open(img_file)
            if args.center_crop_and_resize_to > 0:
                img = resize_and_crop(img, args.center_crop_and_resize_to)
            writer.write({"frame": img2str(img)})
        writer.finalize()

        # save ann
        ann_file = os.path.join(ann_output_dir, f"{uid}.json")
        with open(ann_file, "w") as f:
            json.dump(group, f, indent=2)


def run_jobs(args):
    # global rank
    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    global_rank = job_env.global_rank
    num_tasks = job_env.num_tasks

    # load annotation file
    with open(args.ann_file, "r") as f:
        raw_anns = json.load(f)

    imageid_to_anns = {}
    for ann in raw_anns["annotations"]:
        image_id = ann["image_id"]
        if image_id not in imageid_to_anns:
            imageid_to_anns[image_id] = []
        imageid_to_anns[image_id].append(ann)

    all_groups = []
    curr_group = []
    for idx, img_ann in enumerate(raw_anns["images"]):

        # filter out objects that are not in the center
        h, w = img_ann["height"], img_ann["width"]
        if h > w:
            min_x, max_x = 0, w
            min_y, max_y = (h - w) // 2, (h + w) // 2
        else:
            min_x, max_x = (w - h) // 2, (w + h) // 2
            min_y, max_y = 0, h

        obj_anns = imageid_to_anns[img_ann["id"]]
        obj_anns.sort(key=lambda x: x["area"], reverse=True)
        objects_to_num = {}
        for o in obj_anns:
            x, y, w, h = o["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 < min_x or x1 > max_x or y2 < min_y or y1 > max_y:
                continue
            c = o["category_freeform"]
            if c not in objects_to_num:
                objects_to_num[c] = 0
            objects_to_num[c] += 1

        if not objects_to_num:
            continue

        narration = ""
        for o, n in objects_to_num.items():
            if n == 1:
                narration += f"{o}, "
            else:
                o = o + "s" if not o.endswith("s") else o
                narration += f"{n} {o}, "
        narration = narration[:-2]
        img_ann["narration"] = narration
        curr_group.append(img_ann)
        if (
            len(curr_group) == args.images_per_file
            or idx == len(raw_anns["images"]) - 1
        ):
            all_groups.append(curr_group)
            curr_group = []

    partition_anns = all_groups[global_rank::num_tasks]
    print(f"Rank {global_rank}/{num_tasks} processing {len(partition_anns)} images")

    start_time = time.time()
    save_images_and_anns(args, partition_anns)
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=1,
        tasks_per_node=192,
        cpus_per_task=1,
        slurm_partition="q1",
        name="egoobj",
        mem_gb=1024,
        timeout_min=30,
    )
    job = executor.submit(run_jobs, args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ann_file",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/EgoObjects/EgoObjectsV1_unified_train.json",
        help="Path to the EgoObjects annotation file",
    )
    args.add_argument(
        "--image_root",
        type=str,
        default=f"{DATA_ROOT_DIR}/datasets/EgoObjects/images",
        help="Path to the image files",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/egoobjects",
        help="Path to the output directory",
    )
    args.add_argument(
        "--images_per_file",
        type=int,
        default=2000,
        help="Number of images per arrow file",
    )
    args.add_argument(
        "--center_crop_and_resize_to",
        type=int,
        default=384,
        help="Center crop the frames to a square and resize to this size",
    )
    args = args.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    sys.exit(main(args))
