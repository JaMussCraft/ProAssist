import json
import os
import random
from dataclasses import dataclass
import multiprocessing as mp
import time
from tqdm import tqdm
from mmassist.datasets.prepare.prompts import (
    action_recognition_prompts as system_prompts,
)
from mmassist.model.tokenization_proact import (
    build_tokenizer_and_update_config,
    ProActConfig,
)
from mmassist.configs.arguments import ModelArguments, HfArgumentParser
from mmassist.datasets.prepare.prepare_utils import PreparedSample
from mmassist.configs.arguments import DATA_ROOT_DIR


@dataclass
class PreprocessArgs:
    dataset_name: str = "sthsthv2"
    data_dir: str = f"{DATA_ROOT_DIR}/processed_data/sthsthv2"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/sthsthv2/prepared"
    prefix: str = "narration"
    split: str = "train"  # train or val
    num_proc: int = mp.cpu_count()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, PreprocessArgs))
    model_args, args = parser.parse_args_into_dataclasses()

    # build the tokenizer
    config = ProActConfig(**model_args.to_dict())
    num_tokens_per_img = config.num_tokens_per_img
    use_img_sep_token = model_args.img_sep_token != ""
    tokenizer = build_tokenizer_and_update_config(config)
    chat_formatter = tokenizer.chat_formatter

    all_anns = []
    ann_dir = os.path.join(args.data_dir, "annotations")
    ann_files = sorted(os.listdir(ann_dir))

    def process_video(ann_file: str) -> list[PreparedSample]:
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            video_ann = json.load(f)
        if args.split not in video_ann["split"]:
            return []
        video_uid = video_ann["uid"]
        frame_file_rel = os.path.join("frames", ann_file.replace(".json", ".arrow"))
        frame_file = os.path.join(args.data_dir, frame_file_rel)
        assert os.path.exists(frame_file), f"Frame file {frame_file} does not exist."

        samples = []

        # first system prompt
        sys_prompt = random.choice(system_prompts)
        sys_turn = {"role": "system", "content": sys_prompt}
        formatted = chat_formatter.apply_chat_template([sys_turn])
        sys_turn_len = len(tokenizer.encode(formatted, add_special_tokens=False))
        curr_conv = [sys_turn]
        curr_conv_seq_len = sys_turn_len
        start_idx = 0
        last_end = -1
        for idx, ann in enumerate(video_ann["anns"]):
            # process a narration
            curr_start, curr_end = ann["frame_start_idx"], ann["frame_end_idx"]
            turns = [
                {"role": "frames", "start": curr_start, "end": curr_end},
                {"role": "assistant", "content": ann["label"].capitalize()},
            ]
            formatted = chat_formatter.apply_chat_template(turns)
            turn_len = len(tokenizer.encode(formatted, add_special_tokens=False)) - 1
            # -1 because the first token is the bos token

            if (
                curr_conv_seq_len + turn_len >= model_args.max_seq_len
                or idx == len(video_ann["anns"]) - 1
            ):
                # sanity check for the number of images
                input_string = chat_formatter.apply_chat_template(curr_conv)
                input_ids = tokenizer.encode(
                    input_string, add_special_tokens=False, return_tensors="pt"
                )
                num_img_tokens = (input_ids == tokenizer.img_token_id).sum().item()
                num_imgs = num_img_tokens // num_tokens_per_img
                assert (
                    num_imgs == last_end - start_idx
                ), f"num_imgs: {num_imgs} != {last_end} - {start_idx}"

                assert last_end != -1

                # we are done with the current sample, add it to the list
                sample = PreparedSample(
                    dataset=args.dataset_name,
                    video_uid=video_uid,
                    clip_idx=len(samples),
                    frames_file=frame_file_rel,
                    seq_len=curr_conv_seq_len,
                    max_seq_len=model_args.max_seq_len,
                    num_tokens_per_img=num_tokens_per_img,
                    use_img_sep_token=use_img_sep_token,
                    start_frame_idx=start_idx,
                    end_frame_idx=last_end,
                    conversation=curr_conv,
                    fps=video_ann["fps"],
                )
                samples.append(sample)

                # start a new sample: add a new system prompt and the current caption
                sys_prompt = random.choice(system_prompts)
                sys_turn = {"role": "system", "content": sys_prompt}
                fmtted = chat_formatter.apply_chat_template([sys_turn])
                sys_turn_len = len(tokenizer.encode(fmtted, add_special_tokens=False))
                curr_conv = [sys_turn] + turns
                curr_conv_seq_len = sys_turn_len + turn_len
                start_idx = curr_start
            else:
                # add the caption to the current sample and continue
                curr_conv.extend(turns)
                curr_conv_seq_len += turn_len

            last_end = curr_end

        return samples

    print("Processing...")
    start_time = time.time()
    pool = mp.Pool(args.num_proc)
    jobs = [pool.apply_async(process_video, args=(af,)) for af in ann_files]
    pool.close()

    all_samples = []
    for job in tqdm(jobs):
        all_samples.extend(job.get())
    print(f"Num samples: {len(all_samples)}")

    print("Saving data...")
    os.makedirs(args.output_dir, exist_ok=True)
    use_sep = "+SEP" if use_img_sep_token else ""
    prefix = f"{args.prefix}_{args.split}"
    full_name = f"{prefix}_L{model_args.max_seq_len}_I{num_tokens_per_img}{use_sep}"
    save_file = os.path.join(args.output_dir, f"{full_name}.jsonl")
    with open(save_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time: {elapsed_time:.2f} minutes")
