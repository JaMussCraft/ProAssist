import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import multiprocessing as mp
import time
from tqdm import tqdm
from dataclasses import dataclass
import datasets as hf_datasets

from mmassist.configs.arguments import ModelArguments, HfArgumentParser


from mmassist.datasets.prepare.conversation import (
    split_conversation,
    convert_conversation_time_to_index,
)
from mmassist.datasets.prepare.prompts import (
    proactive_assistant_system_prompts,
    summarize_sys_prompt,
    summarize_query,
)

from mmassist.model.tokenization_proact import (
    build_tokenizer_and_update_config,
    ProActConfig,
)
from mmassist.datasets.prepare.prepare_utils import PreparedSample
from mmassist.configs.arguments import DATA_ROOT_DIR


@dataclass
class DataPrepareArgs:
    dataset: str = "ego4d"
    data_dir: str = f"{DATA_ROOT_DIR}/processed_data"
    split: str = "val"  # train, val, test
    fps: int = 2
    add_knowledge: bool = True
    add_summary: bool = True
    reserved_max_summary_len: int = -1
    summary_only: bool = False
    save_prefix: str = "dialog"
    keep_ctx_length: tuple[int, int] = (5, 20)
    num_proc: int = mp.cpu_count()


if __name__ == "__main__":
    parser = HfArgumentParser((DataPrepareArgs, ModelArguments))
    args, model_args = parser.parse_args_into_dataclasses()
    args.annotation_dir = os.path.join(args.data_dir, args.dataset, "generated_dialogs")
    args.output_dir = os.path.join(args.data_dir, args.dataset, "prepared")

    if args.summary_only:
        if args.add_knowledge:
            args.add_knowledge = False
            print("summary_only=True: disabling add_knowledge.")
        if not args.add_summary:
            args.add_summary = True
            print("summary_only=True: enabling add_summary.")
    if model_args.max_seq_len <= 0:
        if args.add_summary:
            args.add_summary = False
            print("Not limit max_seq_len: disabling add_summary.")

    # config
    config = ProActConfig(**model_args.to_dict())
    num_tokens_per_img = args.num_tokens_per_img = config.num_tokens_per_img
    use_img_sep_token = model_args.img_sep_token != ""
    tokenizer = build_tokenizer_and_update_config(config)
    chat_formatter = tokenizer.chat_formatter

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tuse_img_sep_token: {use_img_sep_token}")
    print(f"\tmax_seq_len: {model_args.max_seq_len}")

    # load the data split
    split = args.split
    with open(os.path.join(args.annotation_dir, f"{split}_filtered.json"), "r") as f:
        data = json.load(f)

    def process_conversation(ann: dict) -> list[dict]:
        try:
            video_uid = ann["video_uid"]
        except:
            print(ann)
            # raise ValueError("video_uid not found in sample")
            return []
        if args.dataset == "ego4d":
            video_uid = video_uid.replace("grp-", "")
        if args.dataset != "assembly101":
            frame_file = f"frames/{video_uid}.arrow"
        else:
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        frame_file_abs = os.path.join(os.path.dirname(args.output_dir), frame_file)
        if not os.path.exists(frame_file_abs):
            print(f"Frame file {frame_file} does not exist.")
            return []

        all_frames_in_video = hf_datasets.Dataset.from_file(frame_file_abs)
        len_frames = len(all_frames_in_video)

        knowledge = ""
        if args.add_knowledge:
            knowledge = f"Task knowledge: {ann['inferred_knowledge']}"
        video_start_time = ann["parsed_video_anns"].get("video_start_time", 0)

        samples: list[PreparedSample] = []
        user_types = set()
        for user_idx, conversation_with_user_info in enumerate(ann["conversations"]):
            try:
                score = conversation_with_user_info["auto_quality_eval"]["final_score"]
            except:
                raise ValueError("auto_quality_eval not found in conversation")
            conversation = conversation_with_user_info["conversation"]

            user_type = conversation_with_user_info["user_type"]
            if args.summary_only and user_type in user_types:
                continue

            max_seq_len = model_args.max_seq_len
            if args.add_summary:
                all_summaries = [t["progress"] for t in conversation if "progress" in t]
                all_summaries.sort(key=lambda x: len(x.split()), reverse=True)
                tokenized = tokenizer(all_summaries[:10], padding=True)
                slen = len(tokenized["input_ids"][0]) + num_tokens_per_img + 20
                # Note: 20 is ~ the length of the summary prompt + the assistant header
                if args.reserved_max_summary_len > args.reserved_max_summary_len:
                    slen = random.randint(slen, args.reserved_max_summary_len)
                if slen > 768:
                    # some conversations have very long summaries, skip them
                    continue
                max_seq_len -= slen

            # convert into the preprocess data format that split_conversation consumes
            converted_conversation = convert_conversation_time_to_index(
                conversation, args.fps, len_frames, video_start_time, knowledge
            )

            num_split_in_curr_conv = 0
            conv_left = converted_conversation
            while conv_left:
                # add some system prompt
                # 1. system prompt should always be added
                if not args.summary_only:
                    system_prompt = random.choice(proactive_assistant_system_prompts)
                else:
                    system_prompt = summarize_sys_prompt
                progress = ""
                if num_split_in_curr_conv > 0:
                    # if it is not the first split
                    # 2. add progress
                    for t in reversed(samples[-1].conversation):
                        if t["role"] == "assistant" and "progress" in t:
                            progress = t["progress"]
                            system_prompt = f"{system_prompt} {progress}"
                            break

                    # 3. add knowledge
                    if args.add_knowledge:
                        system_prompt = f"{system_prompt} {knowledge}"

                conv_left = [{"role": "system", "content": system_prompt}] + conv_left
                if model_args.max_seq_len > 0:
                    splited_conv, conv_left = split_conversation(
                        conversation=conv_left,
                        max_seq_len=max_seq_len,
                        tokenizer=tokenizer,
                        keep_ctx_length=args.keep_ctx_length,
                        fps=args.fps,
                    )
                else:
                    splited_conv = conv_left
                    conv_left = []

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

                    # get the current progress summary
                    progress = None
                    if args.add_summary and end_idx + 1 < len_frames:
                        for i in reversed(splited_conv):
                            if i["role"] == "assistant":
                                progress = i["progress"]
                                break

                    # remove assistant turns to create summary-only samples
                    if args.summary_only:
                        new_splited_conv = []
                        curr_frame_turn = None
                        for turn in splited_conv:
                            if turn["role"] in ["user", "system"]:
                                new_splited_conv.append(turn)
                                curr_frame_turn = None
                            elif turn["role"] == "frames":
                                if curr_frame_turn is None:
                                    curr_frame_turn = turn
                                    new_splited_conv.append(curr_frame_turn)
                                else:
                                    curr_frame_turn["end"] = turn["end"]
                        splited_conv = new_splited_conv

                    # add a progress summary turn
                    if progress is not None:
                        # add a summary prompt
                        splited_conv.append(summarize_query)
                        splited_conv.append(
                            {"role": "frames", "start": end_idx, "end": end_idx + 1}
                        )
                        splited_conv.append({"role": "assistant", "content": progress})
                        end_idx += 1

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

                    # sanity check for the sequence length
                    seq_len = len(input_ids[0])
                    mlen = model_args.max_seq_len
                    assert seq_len <= mlen or mlen <= 0, f"seq_len: {seq_len} > {mlen}"

                    if start_idx != -1 and end_idx != -1 and end_idx < len_frames:
                        user_id = f"{user_type}_{user_idx}"
                        has_summary = progress is not None

                        if args.summary_only and not has_summary:
                            continue

                        metadata = {
                            "user_type": user_type,
                            "user_id": user_id,
                            "task_goal": ann["inferred_goal"],
                            "knowledge": ann["inferred_knowledge"],
                            "progress": progress,
                            "add_knowledge": args.add_knowledge,
                            "has_summary": has_summary,
                            "summary_only": args.summary_only,
                            "quality": score,
                        }
                        prepared_sample = PreparedSample(
                            dataset=ann["parsed_video_anns"]["dataset"],
                            video_uid=video_uid,
                            clip_idx=num_split_in_curr_conv,
                            frames_file=frame_file,
                            seq_len=seq_len,
                            max_seq_len=model_args.max_seq_len,
                            num_tokens_per_img=num_tokens_per_img,
                            use_img_sep_token=use_img_sep_token,
                            start_frame_idx=start_idx,
                            end_frame_idx=end_idx,
                            conversation=splited_conv,
                            fps=args.fps,
                            metadata=metadata,
                        )
                        samples.append(prepared_sample)
                        user_types.add(user_type)
                        num_split_in_curr_conv += 1

        return samples

    print("Processing...")
    start_time = time.time()
    pool = mp.Pool(args.num_proc)
    jobs = [pool.apply_async(process_conversation, args=(dp,)) for dp in data]
    pool.close()

    all_samples = []
    for job in tqdm(jobs):
        all_samples.extend(job.get())
    print(f"Num samples after splitting: {len(all_samples)}")

    prefix = args.save_prefix
    if not args.summary_only:
        if args.add_knowledge:
            prefix += "-klg"
        if args.add_summary:
            prefix += "-sum"
    else:
        prefix = "summary"
    use_sep = "+SEP" if use_img_sep_token else ""
    suffix = f"L{model_args.max_seq_len}_I{num_tokens_per_img}{use_sep}"

    os.makedirs(args.output_dir, exist_ok=True)
    split_file = os.path.join(args.output_dir, f"{prefix}_{split}_{suffix}.jsonl")
    print(f"Split {split} has {len(all_samples)} samples. Saving to {split_file}")
    with open(split_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time: {elapsed_time:.2f} minutes")
