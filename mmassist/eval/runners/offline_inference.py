import os
import json
import torch
from transformers import PreTrainedTokenizer

from mmassist.data.data_collator import ProActCollator
from mmassist.model.modeling_proact import ProActModelMixin, trim_past_key_values
from mmassist.eval.runners.base_runner import BaseInferenceRunner
from mmassist.eval.eval_utils import get_file_path, save_json


@torch.no_grad()
def offline_inference(
    model: ProActModelMixin,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    images: torch.ByteTensor | None = None,
    image_embeds: torch.FloatTensor | None = None,
    not_talk_threshold: float = 0.0,
    max_seq_len: int = 4096,
    **kwargs,
) -> dict:
    """Offline inference for multimodal online video chat.

    The metrics (ppl, frame_diff, fluency, and token_acc) are proposed in the paper:
    "VideoLLM-online: Online Large Language Model for Streaming Video".
    The implementation is based on the official codebase:
    https://github.com/showlab/videollm-online/blob/main/models/modeling_live.py#L45

    On top of them, we add the generative evaluation for each assistant turn given
    the ground-truth context and the start-to-talk signal. This is useful for
    computing NLG metrics (e.g., BLEU, CIDEr, METEOR) for evaluating the quality
    of assistant's responses.
    """
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only support batch_size=1 for now."

    # pre-compute some constants
    device = input_ids.device
    zero = torch.tensor(0, dtype=torch.int, device=device)
    img_token_id = model.config.img_token_id
    img_sep_token_id = model.config.img_sep_token_id
    num_tokens_per_img = model.config.num_tokens_per_img
    eos_token_id = model.config.eos_token_id
    w2t_target_id = model.config.w2t_target_id

    ### debug
    is_debug = False
    tokenizer = kwargs.get("tokenizer", None)
    conversation = kwargs.get("conversation", None)
    if tokenizer is not None and conversation is not None:
        is_debug = True
        assistant_turns = []
        curr_turns = {"inputs": []}
        for t in conversation:
            if t["role"] == "assistant":
                curr_turns["ref"] = t["content"]
                assistant_turns.append(curr_turns)
                curr_turns = {"inputs": []}
            else:
                curr_turns["inputs"].append(t)

    # forward the full input_ids and labels, get tokenwise logits and losses
    outputs = model.forward(
        input_ids=input_ids,
        images=images,
        image_embeds=image_embeds,
        return_dict=True,
        use_cache=True,
        trunc_seq_len=max_seq_len,
        **kwargs,
    )
    logits, past_key_values = outputs.logits, outputs.past_key_values

    if input_ids.shape[1] > max_seq_len:
        if is_debug:
            print(f"Truncating from {input_ids.shape[1]} to {max_seq_len}")
        input_ids = input_ids[:, :max_seq_len]
        labels = labels[:, :max_seq_len]
    if is_debug:
        print(f"input_ids shape: {input_ids.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"logits shape: {logits.shape}")

    # compute the stop position of each assistant turn
    txt_input_mask = input_ids != img_token_id
    non_eos_mask = input_ids != eos_token_id
    pred_eos_mask = labels == eos_token_id
    ass_stop_pos_masks = txt_input_mask & non_eos_mask & pred_eos_mask

    past_num_frames = 0  # global frame index in the batch
    input_id = input_ids[0]
    label = labels[0]
    logit = logits[0]

    # get the start, stop position of each turn:
    # <img> ... assistant: xxx yyy (end-of-turn)
    # (start-of-next-turn)
    ass_stop_pos_mask = ass_stop_pos_masks[0]
    turn_stops = (ass_stop_pos_mask.nonzero() + 1)[:, 0].tolist()
    turn_starts = [0] + turn_stops[:-1]
    num_turns = len(turn_starts)

    # compute metrics for each turn
    lm_ppls, frame_diffs, fluencies, token_acc = [], [], [], []
    generations = []
    num_assistant_turns = 0
    for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
        ## prepare corresponding mask according two losses
        turn_label = label[turn_start:turn_stop]
        turn_learn_mask = turn_label != model.config.ignore_id
        turn_input_id = input_id[turn_start:turn_stop]
        turn_img_mask = turn_input_id == model.config.img_token_id
        turn_num_frames = turn_img_mask.sum() // num_tokens_per_img
        if not turn_learn_mask.any():
            # user/system turn
            past_num_frames += turn_num_frames
            continue
        turn_logit = logit[turn_start:turn_stop]
        turn_pred_mask = turn_img_mask & turn_learn_mask
        turn_lm_mask = turn_learn_mask & ~turn_pred_mask

        ## ppl and token_acc,
        if turn_lm_mask.any():
            # compute ppl
            turn_lm_masked_logit = turn_logit[turn_lm_mask]
            turn_lm_masked_label = turn_label[turn_lm_mask]
            lm_ppl = torch.nn.functional.cross_entropy(
                turn_lm_masked_logit, turn_lm_masked_label
            ).exp()
            lm_ppls.append(lm_ppl)
            turn_lm_masked_wrong_mask = (
                turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
            )
            if turn_lm_masked_wrong_mask.any():
                num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0, 0]
            else:
                num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
            token_acc.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

        if turn_pred_mask.any():
            ## frame_diff (will be casted to time_diff in compute_metrics)

            turn_last_pred_idx = turn_pred_mask.nonzero()[-1, 0]
            start_talk_idx = turn_start + turn_last_pred_idx + 1
            if is_debug:
                start_talk_idx = turn_start + turn_last_pred_idx + 1
                assert (
                    tokenizer.decode(input_id[start_talk_idx]) == "<|start_header_id|>"
                )
                assert tokenizer.decode(label[start_talk_idx]) == "assistant"
            past_key_values_before_talk = trim_past_key_values(
                past_key_values, 0, start_talk_idx
            )

            turn_score = turn_logit.softmax(dim=-1)
            turn_pred_masked_score = turn_score[turn_pred_mask]
            if not_talk_threshold > 0:
                lower_threshold_mask = (
                    turn_pred_masked_score[:, w2t_target_id] < not_talk_threshold
                )
                turn_pred_masked_score[lower_threshold_mask] = 0
            turn_pred_masked_pred_mask = (
                turn_pred_masked_score.argmax(dim=-1, keepdim=True) != w2t_target_id
            )
            if turn_pred_masked_pred_mask.any():
                ## reply before (at) turn_num_frames
                first_pred = turn_pred_masked_pred_mask.nonzero()[0, 0]
                frame_diff = turn_pred_mask.sum() - first_pred - 1
            else:
                ## the most complex part, reply after turn_num_frames.
                # we assume the 'assistant: ...' does not exist

                if r == num_turns - 1:
                    # no future frame
                    frame_diff = zero
                else:
                    next_turn_ids = input_id[turn_starts[r + 1] : turn_stops[r + 1]]
                    next_turn_num_frames = (
                        next_turn_ids == img_token_id
                    ).sum() // num_tokens_per_img
                    to_append_num_frames = min(
                        next_turn_num_frames, turn_num_frames - 1
                    )  # avoid bias. current as center, two equal left/right side
                    if to_append_num_frames == 0:
                        frame_diff = zero
                    else:
                        multi_imgs_maybe_with_sep = []
                        for i in range(to_append_num_frames):
                            img_tokens = [img_token_id] * num_tokens_per_img
                            multi_imgs_maybe_with_sep.extend(img_tokens)
                            if (
                                img_sep_token_id is not None
                                and i < to_append_num_frames - 1
                            ):
                                multi_imgs_maybe_with_sep.append(img_sep_token_id)
                        to_append_input_id = torch.tensor(
                            multi_imgs_maybe_with_sep, dtype=torch.long, device=device
                        )

                        curr_frame_idx = past_num_frames + turn_num_frames

                        if image_embeds is not None:
                            curr_feat_idx = curr_frame_idx * num_tokens_per_img
                            increment = to_append_num_frames * num_tokens_per_img
                            input_embeds = image_embeds[
                                curr_feat_idx : curr_feat_idx + increment
                            ]
                            mminput = {"image_embeds": input_embeds}
                        else:
                            input_imgs = images[
                                curr_frame_idx : curr_frame_idx + to_append_num_frames
                            ]
                            mminput = {"images": input_imgs}

                        to_append_logit = model.forward(
                            input_ids=to_append_input_id[None],
                            past_key_values=past_key_values_before_talk,
                            return_dict=True,
                            use_cache=True,
                            **mminput,
                        ).logits[0]
                        # we only use the last idx of each frame
                        sep_token_offset = 1 if img_sep_token_id is not None else 0
                        end_of_img_pos = torch.arange(
                            num_tokens_per_img - 1,
                            len(to_append_input_id),
                            num_tokens_per_img + sep_token_offset,
                            device=device,
                        )
                        to_append_score = to_append_logit[end_of_img_pos].softmax(
                            dim=-1
                        )
                        if not_talk_threshold > 0:
                            lower_threshold_mask = (
                                to_append_score[:, w2t_target_id] < not_talk_threshold
                            )
                            to_append_score[lower_threshold_mask] = 0
                        to_append_score_pred_mask = (
                            to_append_score.argmax(dim=-1) != w2t_target_id
                        )
                        if to_append_score_pred_mask.any():
                            first_pred = to_append_score_pred_mask.nonzero()[0, 0]
                            frame_diff = -(first_pred + 1)
                        else:
                            frame_diff = -to_append_num_frames
            frame_diffs.append(frame_diff.abs())

            ## text generation
            if turn_lm_mask.any():
                prompt_input_ids = input_id[None, start_talk_idx : start_talk_idx + 1]
                prompt_input_embeds = model.joint_embed(input_ids=prompt_input_ids)
                output_ids, _ = model.fast_greedy_generate(
                    prompt_input_embeds,
                    past_key_values_before_talk,
                    max_length=1024,
                    **kwargs,
                )
                generations.append(output_ids.cpu())

        ## fluency
        if turn_lm_mask.any() and turn_pred_mask.any():
            num_pred_tokens = turn_pred_mask.sum()
            num_pred_and_txt_tokens = num_pred_tokens + turn_lm_masked_label.numel()
            if frame_diff == 0:
                num_correct_tokens = num_pred_tokens + num_lm_correct_tokens
                fluency = num_correct_tokens / num_pred_and_txt_tokens
            elif frame_diff > 0:
                fluency = (num_pred_tokens - frame_diff) / num_pred_and_txt_tokens
            else:
                fluency = (num_pred_tokens - 1) / num_pred_and_txt_tokens
            fluencies.append(fluency)

        # debug
        if is_debug:
            gt_assistant_turn = assistant_turns[num_assistant_turns]
            gt_decoded_utt = tokenizer.decode(turn_input_id[turn_lm_mask])
            gt_decoded_utt = tokenizer.chat_formatter.cleanup_text(gt_decoded_utt)[0]
            print(num_assistant_turns)
            assert gt_assistant_turn["ref"] == gt_decoded_utt
            print(f"GT (REF): {gt_assistant_turn['ref']}")
            print(f"GT (DEC): {gt_decoded_utt}")
            decoded = tokenizer.decode(output_ids[0])
            print(f"GENERATE: {tokenizer.chat_formatter.cleanup_text(decoded)[0]}")

        past_num_frames += turn_num_frames
        num_assistant_turns += 1

    lm_ppl = torch.stack(lm_ppls).mean().item() if lm_ppls else 1.0
    frame_diff = torch.stack(frame_diffs).float().mean().item() if frame_diffs else 0.0
    fluency = torch.stack(fluencies).float().mean().item() if fluencies else 1.0
    token_acc = torch.stack(token_acc).float().mean().item() if token_acc else 1.0

    results = {
        "lm_ppl": lm_ppl,
        "frame_diff": frame_diff,
        "fluency": fluency,
        "token_acc": token_acc,
        "generations": generations,
    }
    return results


class OfflineInferenceRunner(BaseInferenceRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collator = ProActCollator(self.tokenizer, self.chat_formatter)

    @classmethod
    def build(cls, **kwargs) -> "OfflineInferenceRunner":
        return super().build(**kwargs)

    def run_inference_on_video(
        self,
        video: dict,
        output_dir: str = "",
        eval_max_seq_len: int | None = None,
        is_debug: bool = False,
        **kwargs,
    ) -> dict:
        """Run inference on a video conversation.

        :param video: The video conversation data.
        :param output_dir: The directory to save the predictions and metadata.
            Default: Do not save the outputs.
        :param kwargs: Additional arguments to pass to the model.

        :return: The predictions from the model.
        """

        batch = self.collator([video])

        if eval_max_seq_len is None:
            eval_max_seq_len = self.eval_max_seq_len
        assert eval_max_seq_len > 0

        seq_len = len(batch["input_ids"][0])
        overflow = seq_len > eval_max_seq_len

        for k, v in batch.items():
            if isinstance(v, torch.FloatTensor):
                batch[k] = v.to(self.model.device, self.model.dtype)
            else:
                batch[k] = v.to(self.model.device)
        conversation = video["conversation"]

        if is_debug:
            kwargs["tokenizer"] = self.tokenizer
            kwargs["conversation"] = conversation

        # run the model
        predictions = offline_inference(
            self.model, **batch, max_seq_len=eval_max_seq_len, **kwargs
        )
        gen_turn_idx = 0
        turns_without_gen = 0
        generations = predictions.pop("generations")
        for turn in conversation:
            if turn["role"] == "assistant":
                if gen_turn_idx < len(generations):
                    gen = generations[gen_turn_idx][0]
                    txt = self.tokenizer.decode(gen)
                    cleaned_txt = self.chat_formatter.cleanup_text(txt)[0]
                    turn["gen"] = cleaned_txt
                    gen_turn_idx += 1
                else:
                    turns_without_gen += 1
                    assert overflow, f"length: {seq_len} | conv: {conversation},"
                    break
        predictions["conversation"] = conversation
        predictions["overflow"] = overflow
        turns_with_gen_ratio = gen_turn_idx / (gen_turn_idx + turns_without_gen)
        predictions["turn_with_gen_ratio"] = turns_with_gen_ratio

        if output_dir:
            save_file = get_file_path(output_dir, video["sample_idx"], "json")
            save_json(predictions, save_file)

        return predictions

    @staticmethod
    def load_predictions(file: str) -> dict:
        """Load the predictions from a file."""
        return json.load(open(file, "r"))
