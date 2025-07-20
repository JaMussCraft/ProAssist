#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import torch
import submitit
import torch
import argparse
import datasets as hf_datasets
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from mmassist.data.utils import img_base64_to_tensor
from mmassist.model.vision import ProActConfig, VisualEncoder, adaptive_avg_pool2d
from mmassist.configs.arguments import DATA_ROOT_DIR

class Task:
    def __call__(self, args):
        job_env = submitit.JobEnvironment()
        local_rank = job_env.local_rank
        global_rank = job_env.global_rank
        num_tasks = job_env.num_tasks

        # Actual task / computation
        # Load the model
        layer = args.img_patch_token_layer
        patch_sizes = [int(p) for p in args.extract_patch_layouts.split(",")]
        config = ProActConfig(
            vision_pretrained=args.vision_pretrained,
            use_img_cls_token=True,
            img_patch_token_layer=layer,
            img_patch_token_size=-1,  # keep all the patches
        )
        print("Config: ", config.to_dict())
        model = VisualEncoder.from_config(config)
        model = model.to(f"cuda:{local_rank}", torch.float16)
        model.eval()
        print("Model loaded")

        all_frames_files = sorted(os.listdir(args.preprocessed_frames_dir))
        frames_files_for_this_rank = all_frames_files[global_rank::num_tasks]
        print(f"Rank {global_rank} processing {len(frames_files_for_this_rank)} files")
        print(frames_files_for_this_rank)

        model_name_str = f"{args.vision_pretrained.replace('/', '___')}@{layer}"
        out_dir = os.path.join(args.output_dir, model_name_str)
        print("Saving to:", out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for file in tqdm(frames_files_for_this_rank):
            frames_file = os.path.join(args.preprocessed_frames_dir, file)
            frames_data = hf_datasets.load_dataset(
                "arrow", data_files=frames_file, split="train"
            )
            out_file = os.path.join(out_dir, file)
            writer = ArrowWriter(path=out_file)
            with torch.no_grad():
                for batch_idx in range(0, len(frames_data), args.batch_size):
                    batch = frames_data[batch_idx : batch_idx + args.batch_size]
                    frames = batch["frame"]
                    imgs = torch.stack([img_base64_to_tensor(d) for d in frames])
                    out = model.encode(imgs)
                    batch_cls_feats, batch_patch_feats = out[:, :1], out[:, 1:]

                    batch_feats = {"cls": batch_cls_feats.cpu().half()}
                    for p in patch_sizes:
                        patch_feats = adaptive_avg_pool2d(batch_patch_feats, (p, p))
                        batch_feats[f"{p}x{p}"] = patch_feats.cpu().half()

                    for i in range(len(frames)):
                        feats = {}
                        for k, v in batch_feats.items():
                            feats[k] = v[i]
                        writer.write(feats)
            writer.finalize()


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=8,
        tasks_per_node=8,
        gpus_per_node=8,
        cpus_per_task=24,
        slurm_partition="q1",
        name="encode_frames",
        mem_gb=1024,
        timeout_min=60 * 24,
        # account="ar-ai-research-interns",
        # account="ar-ai-midpri",
    )
    job = executor.submit(Task(), args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--preprocessed_frames_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/llava/frames",
        help="Path to the preprocessed video frames directory",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_ROOT_DIR}/processed_data/llava/features",
        help="Path to save the extracted features",
    )
    args.add_argument(
        "--vision_pretrained",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Vision model name",
    )
    args.add_argument(
        "--img_patch_token_layer",
        type=int,
        default=-2,
        help="which layer to extract image features from",
    )
    args.add_argument(
        "--extract_patch_layouts",
        type=str,
        default="2,3,5",
        help="Comma separated list of patch sizes to extract features",
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for encoding frames",
    )

    args = args.parse_args()

    print(args)

    sys.exit(main(args))
