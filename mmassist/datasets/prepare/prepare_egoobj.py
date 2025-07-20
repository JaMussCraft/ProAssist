import json
import os
import random
from dataclasses import dataclass
import multiprocessing as mp
import time
from tqdm import tqdm
from mmassist.datasets.prepare.prompts import (
    object_detection_prompts as system_prompts,
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
    dataset_name: str = "egoobjects"
    data_dir: str = f"{DATA_ROOT_DIR}/processed_data/egoobjects"
    output_dir: str = f"{DATA_ROOT_DIR}/processed_data/egoobjects/prepared"
    prefix: str = "detection_train"
    num_proc: int = mp.cpu_count()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, PreprocessArgs))
    model_args, args = parser.parse_args_into_dataclasses()
    model_args.img_sep_token = ""  # no image separator token

    # build the tokenizer
    config = ProActConfig(**model_args.to_dict())
    num_tokens_per_img = config.num_tokens_per_img
    use_sep_token = model_args.img_sep_token != ""
    tokenizer = build_tokenizer_and_update_config(config)
    chat_formatter = tokenizer.chat_formatter

    all_anns = []
    ann_dir = os.path.join(args.data_dir, "annotations")
    ann_files = sorted(os.listdir(ann_dir))

    def process_video(ann_file: str) -> list[PreparedSample]:
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            anns = json.load(f)
        vid = ann_file.replace(".json", "")
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

        for idx, ann in enumerate(anns):
            # process a caption
            turns = [
                {"role": "frames", "start": idx, "end": idx + 1},
                {"role": "assistant", "content": ann["narration"]},
            ]
            formatted = chat_formatter.apply_chat_template(turns)
            turn_len = len(tokenizer.encode(formatted, add_special_tokens=False)) - 1

            if (
                curr_conv_seq_len + turn_len >= model_args.max_seq_len
                or idx == len(anns) - 1
            ):
                # sanity check for the number of images
                input_string = chat_formatter.apply_chat_template(curr_conv)
                input_ids = tokenizer.encode(
                    input_string, add_special_tokens=False, return_tensors="pt"
                )
                num_img_tokens = (input_ids == tokenizer.img_token_id).sum().item()
                num_imgs = num_img_tokens // num_tokens_per_img
                assert (
                    num_imgs == idx - start_idx
                ), f"num_imgs: {num_imgs} != {idx} - {start_idx}"

                # we are done with the current sample, add it to the list
                sample = PreparedSample(
                    dataset=args.dataset_name,
                    video_uid=vid,
                    clip_idx=len(samples),
                    frames_file=frame_file_rel,
                    seq_len=curr_conv_seq_len,
                    max_seq_len=model_args.max_seq_len,
                    num_tokens_per_img=num_tokens_per_img,
                    use_img_sep_token=use_sep_token,
                    start_frame_idx=start_idx,
                    end_frame_idx=idx,
                    conversation=curr_conv,
                    metadata=ann,
                )
                samples.append(sample)

                # start a new sample: add a new system prompt and the current caption
                sys_prompt = random.choice(system_prompts)
                sys_turn = {"role": "system", "content": sys_prompt}
                fmted = chat_formatter.apply_chat_template([sys_turn])
                sys_turn_len = len(tokenizer.encode(fmted, add_special_tokens=False))
                curr_conv = [sys_turn] + turns
                curr_conv_seq_len = sys_turn_len + turn_len
                start_idx = idx
            else:
                # add the caption to the current sample and continue
                curr_conv.extend(turns)
                curr_conv_seq_len += turn_len

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
    full_name = f"{args.prefix}_L{model_args.max_seq_len}_I{num_tokens_per_img}"
    save_file = os.path.join(args.output_dir, f"{full_name}.jsonl")
    with open(save_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time: {elapsed_time:.2f} minutes")
