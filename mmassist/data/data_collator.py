import random
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from mmassist.model.tokenization_proact import MultimodalChat


@dataclass
class ProActCollator(object):
    tokenizer: PreTrainedTokenizer
    chat_formatter: MultimodalChat
    compute_labels: bool = True
    pose_maxlen: int = 0
    use_binary_decision_head: bool = False

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        texts, batch_learn_ranges = [], []
        for sample in samples:
            conv = sample["conversation"]
            nfsr = sample["neg_frame_sampling_rate"]
            texts.append(self.chat_formatter.apply_chat_template(conv))
            batch_learn_ranges.append(self.chat_formatter.get_learn_ranges(conv, nfsr))

        batch = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False,  # truncate in model's forward method
        )
        batch["sample_idx"] = torch.tensor([s["sample_idx"] for s in samples])

        if "encoded_images" in samples[0]:
            all_embeds = torch.cat([s["encoded_images"].flatten(0, 1) for s in samples])
            batch["image_embeds"] = all_embeds

        if "images" in samples[0]:
            batch["images"] = torch.cat([s["images"] for s in samples])

        if not self.compute_labels:
            batch.pop("offset_mapping")
            batch.pop("attention_mask")
            return batch

        batch_labels = torch.full_like(
            batch.input_ids, self.tokenizer.ignore_id, dtype=torch.long
        )

        for labels, input_ids, offset_mapping, learn_range in zip(
            batch_labels,
            batch.input_ids,
            batch.offset_mapping,
            batch_learn_ranges,
        ):
            last_start = offset_mapping[-1, 0]
            for learn_r in learn_range:
                if learn_r.start > last_start:
                    # this should not happen without truncation
                    break
                start = torch.nonzero(offset_mapping[:, 0] == learn_r.start).item()
                if offset_mapping[:, 0][-1] >= learn_r.stop:
                    stop = torch.nonzero(offset_mapping[:, 0] == learn_r.stop).item()
                else:  # the last eos token
                    stop = len(input_ids)
                labels[start - 1 : stop - 1] = input_ids[start:stop]

        # if separator token is not used, use the eos token as label for predicting
        # whether to talk or not
        if self.tokenizer.img_sep_token_id is None:
            batch_labels[batch_labels == self.tokenizer.img_token_id] = (
                self.tokenizer.eos_token_id
            )
        batch["labels"] = batch_labels

        if self.use_binary_decision_head:
            w2t_id = self.tokenizer.w2t_target_id
            bor_id = self.tokenizer.bor_token_id
            img_tokens_mask = batch.input_ids == self.tokenizer.img_token_id
            batch["neg_frame_mask"] = (batch_labels == w2t_id) & img_tokens_mask
            batch["pos_frame_mask"] = (batch_labels == bor_id) & img_tokens_mask

        # if use PoSE (NOTE: current imply only works for right padding)
        if self.pose_maxlen > 0:
            padded_seq_len = batch.input_ids.size(1)
            position_ids = []
            for input_ids, attn_mask in zip(batch.input_ids, batch.attention_mask):
                curr_len = attn_mask.sum().item()
                pos_ids = torch.arange(padded_seq_len, dtype=torch.long)
                if self.pose_maxlen - curr_len > 0:
                    offset = random.randint(0, self.pose_maxlen - curr_len)
                    split_pos = random.randint(0, curr_len - 1)
                    pos_ids[split_pos:] += offset
                position_ids.append(pos_ids)
            batch["position_ids"] = torch.stack(position_ids)

        batch.pop("attention_mask")  # flash attention cannot use it anyway
        batch.pop("offset_mapping")
        return batch
