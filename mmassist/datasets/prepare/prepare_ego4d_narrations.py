import json
import os
import random
import multiprocessing as mp
import time
from tqdm import tqdm
from dataclasses import dataclass

from mmassist.configs.arguments import ModelArguments, HfArgumentParser
from mmassist.datasets.utils.ego4d_utils import group_by_split
from mmassist.datasets.prepare.conversation import split_conversation
from mmassist.datasets.prepare.prompts import (
    ego4d_narration_system_prompts as system_prompts,
)
from mmassist.model.tokenization_proact import (
    build_tokenizer_and_update_config,
    ProActConfig,
)
from mmassist.datasets.prepare.prepare_utils import PreparedSample
from mmassist.configs.arguments import DATA_ROOT_DIR

ROOT_DIR = f"{DATA_ROOT_DIR}/processed_data/"


@dataclass
class PreprocessArgs:
    dataset_name: str = "ego4d"
    data_dir: str = ROOT_DIR + "ego4d"
    ego4d_data_splits_file: str = ROOT_DIR + "ego4d/data_splits.json"
    our_data_split_file: str = ROOT_DIR + "ego4d/val_test_split.json"
    output_dir: str = ROOT_DIR + "ego4d/prepared"
    prefix: str = "narration"
    num_proc: int = mp.cpu_count()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, PreprocessArgs))
    model_args, args = parser.parse_args_into_dataclasses()

    # build the tokenizer
    print("img_sep_token:", model_args.img_sep_token)
    config = ProActConfig(**model_args.to_dict())
    num_tokens_per_img = args.num_tokens_per_img = config.num_tokens_per_img
    use_img_sep_token = model_args.img_sep_token != ""
    tokenizer = build_tokenizer_and_update_config(config)
    chat_formatter = tokenizer.chat_formatter

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    all_anns = []
    ann_dir = os.path.join(args.data_dir, "processed_narrations")
    ann_files = sorted(os.listdir(ann_dir))

    def process_conversation(ann_file: str) -> list[PreparedSample]:
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            ann = json.load(f)

        frame_file = os.path.join(args.data_dir, ann["frames_file"])
        if not os.path.exists(frame_file):
            print(f"No frames for {ann['video_uid']}: skipping")
            return []

        samples = []
        conv_left = ann["conversation"]
        while conv_left:
            system_prompt = random.choice(system_prompts)
            conv_left = [{"role": "system", "content": system_prompt}] + conv_left
            splited_conv, conv_left = split_conversation(
                conversation=conv_left,
                max_seq_len=model_args.max_seq_len,
                tokenizer=tokenizer,
                keep_ctx_length=(1, 3),
                fps=ann["sample_fps"],
            )

            if splited_conv:
                # find the start and end frame indices in the splited_conv
                start_idx, end_idx = -1, -1
                for i in splited_conv:
                    if i["role"] == "frames":
                        start_idx = i["start"]
                        break
                for i in reversed(splited_conv):
                    if i["role"] == "frames":
                        end_idx = i["end"]
                        break

                # sanity check for the number of images
                input_string = chat_formatter.apply_chat_template(splited_conv)
                input_ids = tokenizer.encode(
                    input_string, add_special_tokens=False, return_tensors="pt"
                )
                num_img_tokens = (input_ids == tokenizer.img_token_id).sum().item()
                num_imgs = num_img_tokens // num_tokens_per_img
                assert (
                    num_imgs == end_idx - start_idx
                ), f"num_imgs: {num_imgs} != {end_idx} - {start_idx}"

                if start_idx != -1 and end_idx != -1:
                    sample = PreparedSample(
                        dataset=args.dataset_name,
                        video_uid=ann["video_uid"],
                        clip_idx=len(samples),
                        frames_file=ann["frames_file"],
                        seq_len=len(input_ids[0]),
                        max_seq_len=model_args.max_seq_len,
                        num_tokens_per_img=num_tokens_per_img,
                        use_img_sep_token=use_img_sep_token,
                        start_frame_idx=start_idx,
                        end_frame_idx=end_idx,
                        conversation=splited_conv,
                        fps=ann["sample_fps"],
                    )
                    samples.append(sample)

        return samples

    print("Processing narrations...")
    start_time = time.time()
    pool = mp.Pool(args.num_proc)
    jobs = [pool.apply_async(process_conversation, args=(af,)) for af in ann_files]
    pool.close()

    all_samples: list[PreparedSample] = []
    for job in tqdm(jobs):
        all_samples.extend(job.get())
    all_samples = [s.to_dict() for s in all_samples]
    print(f"Num samples after splitting: {len(all_samples)}")

    print("Saving the data into splits...")
    split_to_samples = group_by_split(
        args.ego4d_data_splits_file, args.our_data_split_file, all_samples
    )
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix
    use_sep = "+SEP" if use_img_sep_token else ""
    suffix = f"L{model_args.max_seq_len}_I{num_tokens_per_img}{use_sep}"
    for split, split_samples in split_to_samples.items():
        split_file = os.path.join(args.output_dir, f"{prefix}_{split}_{suffix}.jsonl")
        print(f"Split {split} has {len(split_samples)} samples.")
        if split_samples:
            with open(split_file, "w") as f:
                for sample in split_samples:
                    f.write(json.dumps(sample) + "\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time: {elapsed_time:.2f} minutes")
