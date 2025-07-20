#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import json
import torch
import submitit
import torch
import argparse
from transformers import Cache

from mmassist.model import build_from_checkpoint
from mmassist.data import build_train_dataset, ProActCollator
from mmassist.model.modeling_proact import ProActModelMixin

@torch.no_grad()
def fast_greedy_generate(
    model: ProActModelMixin,
    inputs_embeds: torch.Tensor,
    past_key_values: Cache | None,
    eos_token_id: int,
    inplace_output_ids: torch.Tensor,
):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(
            inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = outputs.past_key_values
        # if i == 0:
        #     past_key_values_to_return = past_key_values
        # print(inputs_embeds.shape)
        # print(past_key_values[0][0].shape)
        # outputs.logits[:, -1, eos_token_id] -= 1
        # topk_logits, topk_indices = outputs.logits[:, -1].topk(5, dim=-1)
        # print("top-5", topk_indices, topk_logits)
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, : i + 1], past_key_values  # past_key_values_to_return


def post_process_gen_txt(text: str) -> str:
    text = text.replace("<|eot_id|>", "")
    if not text:
        return ""
    if "\n\n" in text:
        text = text.split("\n\n")[1]
    return text


def trim_past_key_values(past_key_values, start, stop, batch_idx: int = -1):
    if batch_idx == -1:
        return tuple(
            [
                (past_keys[:, :, start:stop], past_values[:, :, start:stop])
                for past_keys, past_values in past_key_values
            ]
        )
    else:
        b = batch_idx
        return tuple(
            [
                (
                    past_keys[b : b + 1, :, start:stop],
                    past_values[b : b + 1, :, start:stop],
                )
                for past_keys, past_values in past_key_values
            ]
        )


class Task:
    def __call__(self, args):
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        rank = dist_env.rank
        num_worlds = dist_env.world_size

        # Actual task / computation
        # Load the model
        model, tokenizer = build_from_checkpoint(args.model_path, is_training=False)
        model = model.to(f"cuda:{dist_env.local_rank}")
        model.eval()
        print("Model loaded")

        config = model.config

        all_args_dict = config.to_dict()
        all_args_dict.update(config.training_args)
        all_args_dict["train_datasets"] = all_args_dict["eval_datasets"]
        dataset = build_train_dataset(**all_args_dict)
        print("Evaluation dataset:", dataset.datasets[0].data_file)
        # dataset.datasets[0].img_feature_dir = ""

        chat_formatter = tokenizer.chat_formatter
        collator = ProActCollator(tokenizer, chat_formatter)

        out_dir = os.path.join(config.training_args["output_dir"], "generated_text")
        print("Saving to:", out_dir)
        os.makedirs(out_dir, exist_ok=True)

        for sample_idx in range(rank, len(dataset), num_worlds):
            print(f"Processing sample {sample_idx} / {len(dataset)}")
            video = dataset[sample_idx]
            inputs = {
                k: v.to(f"cuda:{dist_env.local_rank}")
                for k, v in collator([video]).items()
            }

            with torch.no_grad():
                input_embeds = model.joint_embed(**inputs)
                inputs["inputs_embeds"] = input_embeds
                outputs = model(**inputs, use_cache=True, return_dict=True)

            tolerance = args.tolerance

            eos_token_id = model.config.eos_token_id
            img_token_id = model.config.img_token_id
            img_sep_token_id = model.config.img_sep_token_id
            img_sep_token_emb = model.get_input_embeddings()(
                torch.tensor(img_sep_token_id).to(f"cuda:{dist_env.local_rank}")
            )[None, None]
            num_tokens_per_img = model.config.num_tokens_per_img
            num_tokens_per_img_with_sep = num_tokens_per_img + 1

            labels = inputs["labels"]
            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            asst_first_token_id = tokenizer.encode(chat_formatter.bor)[-1]

            idx_in_batch = 0
            turn_start_token_mask = labels[idx_in_batch] == asst_first_token_id
            asst_utt_start_pos = set(
                turn_start_token_mask.nonzero()[:, idx_in_batch].cpu().tolist()
            )
            this_input_ids = inputs["input_ids"][idx_in_batch]
            this_input_embeds = input_embeds[idx_in_batch]
            this_logits = logits[idx_in_batch]
            this_preds = preds[idx_in_batch]

            inplace_output_ids = torch.full(
                (1, 100), -100, dtype=torch.long, device=f"cuda:{dist_env.local_rank}"
            )

            # 1. prepare multi-turn start and stop
            turn_stops = ((this_input_ids == eos_token_id).nonzero() + 1)[:, 0].tolist()
            turn_starts = [0] + turn_stops[:-1]

            all_gt_texts = []
            all_gen_texts = []
            result = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

            for tidx, (turn_start, turn_stop) in enumerate(
                zip(turn_starts, turn_stops)
            ):
                if turn_start not in asst_utt_start_pos:
                    label = labels[idx_in_batch, turn_start]
                    pred = preds[idx_in_batch, turn_start]
                    if label == eos_token_id:
                        if pred != eos_token_id:
                            result["FP"] += 1
                        else:
                            result["TN"] += 1
                    continue

                gt_assist_text = tokenizer.decode(this_input_ids[turn_start:turn_stop])
                all_gt_texts.append(gt_assist_text)
                gen_texts = []

                for i in range(-tolerance, 1):
                    pred_pos = turn_start + i * num_tokens_per_img_with_sep
                    if pred_pos < 0:
                        break
                    if this_input_ids[pred_pos] != img_token_id:
                        continue

                    if preds[idx_in_batch, pred_pos] == tokenizer.eos_token_id:
                        # prediction will be an eos token
                        continue
                    else:
                        # make a prediction here
                        kv_cache = trim_past_key_values(
                            outputs.past_key_values, 0, pred_pos, idx_in_batch
                        )
                        output_ids, _ = fast_greedy_generate(
                            model,
                            inputs_embeds=this_input_embeds[
                                None, pred_pos : pred_pos + 1
                            ],
                            past_key_values=kv_cache,
                            eos_token_id=tokenizer.eos_token_id,
                            inplace_output_ids=inplace_output_ids,
                        )
                        gen_text = tokenizer.decode(output_ids[0])
                        gen_texts.append(gen_text)
                        break

                if gen_texts:
                    result["TP"] += 1
                    all_gen_texts.append(gen_texts)
                    continue

                for i in range(1, tolerance + 1):
                    try:
                        future_turn_start, future_turn_stop = (
                            turn_starts[tidx + i],
                            turn_stops[tidx + i],
                        )
                    except:
                        break
                    if future_turn_start in asst_utt_start_pos:
                        break
                    pred_pos = future_turn_stop - 1

                    # make a prediction here

                    # include the kv cache to the frame token before generating the gt utterance
                    kv_cache = trim_past_key_values(
                        outputs.past_key_values, 0, turn_start + 1, idx_in_batch
                    )
                    # include the input embeds of the next i frames
                    append_input_embeds = torch.cat(
                        [
                            img_sep_token_emb,
                            this_input_embeds[turn_stop:pred_pos][None, :],
                        ],
                        dim=1,
                    )
                    output_ids, _ = fast_greedy_generate(
                        model,
                        inputs_embeds=append_input_embeds,
                        past_key_values=kv_cache,
                        eos_token_id=tokenizer.eos_token_id,
                        inplace_output_ids=inplace_output_ids,
                    )
                    gen_text = tokenizer.decode(output_ids[0])
                    if output_ids[0, 0] != model.config.img_sep_token_id:
                        gen_texts.append(gen_text)
                        break

                if gen_texts:
                    result["TP"] += 1
                else:
                    result["FN"] += 1
                all_gen_texts.append(gen_texts)

                # if tidx > 10:
                #     break

            result_dict = {
                "sample_idx": sample_idx,
                "video_id": video["video_id"],
                "result": result,
                "gt_texts": all_gt_texts,
                "gen_texts": all_gen_texts,
            }

            with open(os.path.join(out_dir, f"{sample_idx}.json"), "w") as f:
                json.dump(result_dict, f)


def main(args):
    executor = submitit.AutoExecutor(folder="slurm_logs/%j")
    executor.update_parameters(
        nodes=8,
        tasks_per_node=8,
        gpus_per_node=8,
        cpus_per_task=24,
        slurm_partition="q1",
        # account="ar-ai-research-interns",
        name="eval",
        mem_gb=1024,
        timeout_min=60 * 24,
    )
    task = Task()
    job = executor.submit(task, args)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_path",
        type=str,
    )
    args.add_argument(
        "--tolerance",
        type=int,
        default=1,
        help="window size",
    )
    args = args.parse_args()

    sys.exit(main(args))
