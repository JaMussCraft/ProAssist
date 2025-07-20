import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM, Cache
from transformers.utils import logging

from mmassist.model.configuration_proact import ProActLlamaConfig, ExceedContextHandling
from mmassist.model.vision import VisualEncoder

logger = logging.get_logger(__name__)

KV_CACHE = tuple[tuple[torch.FloatTensor, torch.FloatTensor]]

ce_loss = nn.functional.cross_entropy
bce_loss = nn.functional.binary_cross_entropy_with_logits


class ProActModelMixin(AutoModelForCausalLM):

    def _init_multimodal_modules(self, mm_feature_size, lm_input_size) -> None:
        self.vision_encoder: VisualEncoder | None = None
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_feature_size, lm_input_size, bias=True),
            nn.GELU(),
            nn.Linear(lm_input_size, lm_input_size, bias=True),
        )
        self.mm_projector.to(self.device, self.dtype)

        # add a binary decision head for whether to talk
        self.binary_decision_head: nn.Module | None = None
        if self.config.use_binary_decision_head:
            hs = self.config.hidden_size
            if "linear" in self.config.binary_decision_head_type:
                self.binary_decision_head = nn.Linear(hs, 1)
            else:
                # hs -> hs//2 -> 1
                self.binary_decision_head = nn.Sequential(
                    nn.Linear(hs, hs // 2),
                    nn.GELU(),
                    nn.Linear(hs // 2, 1),
                )
            self.binary_decision_head.to(self.device, self.dtype)

    def mm_feature_proj(self, features: torch.Tensor) -> torch.Tensor:
        assert self.mm_projector is not None, "Please init multimodal projector first"
        return self.mm_projector(features)

    def init_vision_encoder(self) -> None:
        logger.warning_once(
            "!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction."
        )
        self.vision_encoder = VisualEncoder.from_config(self.config)
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.to(self.device, self.get_input_embeddings().weight.dtype)

    def visual_embed(self, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        if self.vision_encoder is None:
            self.init_vision_encoder()
        assert self.vision_encoder is not None
        # images: Nx3xHxW
        with torch.no_grad():
            vis_feats = []
            for b in range(0, len(images), batch_size):
                feats = self.vision_encoder.encode(images[b : b + batch_size])
                vis_feats.append(feats.view(-1, feats.shape[-1]))
            vis_feats = torch.cat(vis_feats)
        return vis_feats

    def joint_embed(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if images is None and image_embeds is None:
            clamped_input_ids = input_ids.clamp(max=self.vocab_size - 1)
            return self.get_input_embeddings()(clamped_input_ids)
        elif image_embeds is None:
            image_embeds = self.visual_embed(images)
        image_embeds = self.mm_feature_proj(image_embeds.to(self.dtype))

        clamped_input_ids = input_ids.clamp(max=self.vocab_size - 1)
        # Note: since we did not resize the input embedding for the added image token
        inputs_embeds = self.get_input_embeddings()(clamped_input_ids)
        inputs_embeds[input_ids == self.config.img_token_id] = image_embeds
        return inputs_embeds

    @torch.no_grad()
    def fast_greedy_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Cache | None,
        max_length: int = 100,
        drop_generated_kv_cache: bool = False,
        not_talk_threshold: float | str = 0.5,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, KV_CACHE]:
        """

        :param inputs_embeds:
        :param past_key_values:
        :param max_length:
        :param drop_generated_kv_cache: if True, the generated key values will be
            dropped. This is useful when use the ground truth context.
        :param not_talk_threshold: the model will keep silent only if the probability of
            predicting the not-to-talk token is higher than this threshold. If a string
            is provided, it should be in the format of "diff0.5" where 0.5 is the threshold
            where the difference between the current and the last not-to-talk probability
            is used to determine whether to talk.
        :return:
        """
        if (
            not hasattr(self, "inplace_output_ids")
            or self.inplace_output_ids.shape[1] < max_length
        ):
            self.inplace_output_ids = torch.full(
                (1, max_length), -100, dtype=torch.long, device=self.device
            )

        past_key_values_to_return = past_key_values
        not_talk_id = self.config.w2t_target_id
        for i in range(max_length):
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            # topk_logits, topk_indices = outputs.logits[:, -1].topk(5, dim=-1)
            # print(f"{i} top-5", topk_indices, topk_logits)
            if i == 0:
                past_key_values_to_return = past_key_values
                # probs = outputs.logits.softmax(dim=-1)[0]
                # not_talk_id = self.config.w2t_target_id
                # not_talk_prob = probs[-1, not_talk_id]
                not_talk_prob = outputs.w2t_probs[0, -1]
                if verbose:
                    print(f"Not talk prob: {not_talk_prob.item()}")

                if (
                    isinstance(not_talk_threshold, str)
                    and self.last_not_talk_prob is not None
                ):
                    diff = self.last_not_talk_prob - not_talk_prob
                    threshold = float(not_talk_threshold[4:])
                    if diff > threshold:
                        if verbose:
                            print(f"diff: {diff:.2f} prob: {not_talk_prob:.2f} talk!")
                        not_talk_threshold = 1.0
                    else:
                        not_talk_threshold = 0.5

                if not_talk_threshold > 0:
                    v = 1e4 if not_talk_prob > not_talk_threshold else -1e4
                    outputs.logits[:, -1, not_talk_id] = v

                self.last_not_talk_prob = not_talk_prob

                if not_talk_threshold > 0:
                    v = 1e4 if not_talk_prob > not_talk_threshold else -1e4
                    outputs.logits[:, -1, not_talk_id] = v
            new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            self.inplace_output_ids[:, i] = new_token_id
            if new_token_id == self.config.eos_token_id or (
                i == 0 and new_token_id == not_talk_id
            ):
                break
            inputs_embeds = self.get_input_embeddings()(new_token_id)

        if not drop_generated_kv_cache:
            past_key_values_to_return = past_key_values
        return self.inplace_output_ids[:, : i + 1], past_key_values_to_return

    def reset_init_kv_cache(self) -> None:
        self.init_key_values_to_keep = None

    def set_init_kv_cache(self, past_key_values: KV_CACHE) -> None:
        self.init_key_values_to_keep = past_key_values


class ProActLlamaForCausalLM(LlamaForCausalLM, ProActModelMixin):
    config_class = ProActLlamaConfig
    _keys_to_ignore_on_load_missing = ["vision_encoder", "mm_projector"]

    def __init__(self, config: ProActLlamaConfig) -> None:
        super().__init__(config)
        self.last_not_talk_prob = None

    def init_multimodal_modules(self) -> None:
        self._init_multimodal_modules(
            self.config.vision_hidden_size, self.config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        images: torch.IntTensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: KV_CACHE | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        neg_frame_mask: torch.BoolTensor | None = None,
        pos_frame_mask: torch.BoolTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        trunc_seq_len: int | None = None,
        **kwargs,
    ):
        output_hidden_states = (
            self.config.use_binary_decision_head or output_hidden_states
        )
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, images, image_embeds)

        # Truncate inputs_embeds if it is too long for training
        if self.training and self.config.max_seq_len > 0:
            trunc_seq_len = self.config.max_seq_len

        if trunc_seq_len is not None and inputs_embeds.shape[1] > trunc_seq_len:
            # Note: we have to do the truncation here instead of in the tokenizer
            # because otherwise the number of image tokens will mismatch the number
            # of image features
            # Note: this should not happen in practice, as we have already
            # preprocessed our data within the max_seq_len
            inputs_embeds = inputs_embeds[:, :trunc_seq_len]
            if input_ids is not None:
                input_ids = input_ids[:, :trunc_seq_len]
            if labels is not None:
                labels = labels[:, :trunc_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :trunc_seq_len]
            if position_ids is not None:
                position_ids = position_ids[:, :trunc_seq_len]

        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # labels
            use_cache=use_cache if not self.training else False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if self.config.use_binary_decision_head:
            assert self.binary_decision_head is not None
            last_hidden_state = outputs.hidden_states[-1]
            w2t_logits = self.binary_decision_head(last_hidden_state)
            outputs.w2t_logits = w2t_logits.squeeze(-1)
            if not self.training:
                outputs.w2t_probs = w2t_logits.sigmoid()
        else:
            if not self.training:
                w2t_id = self.config.w2t_target_id
                outputs.w2t_logits = None
                outputs.w2t_probs = outputs.logits.softmax(dim=-1)[:, :, w2t_id]

        loss = None
        log_dict = {}
        if labels is not None:
            logits = outputs.logits
            weights = None
            if not self.config.use_binary_decision_head:
                if self.config.w2t_logit_weight != 1:
                    if neg_frame_mask is not None:
                        mask = neg_frame_mask.flatten()
                    else:
                        w2t_id = self.config.w2t_target_id
                        img_tokens_mask = input_ids == self.tokenizer.img_token_id
                        mask = (labels == w2t_id) & img_tokens_mask
                    weights = mask * self.config.w2t_logit_weight + ~mask
            else:
                labels[neg_frame_mask] = self.config.ignore_id

            if weights is None:
                loss = ce_loss(logits.flatten(0, 1), labels.flatten())
            else:
                loss = ce_loss(logits.flatten(0, 1), labels.flatten(), reduction="none")
                loss = loss * weights
                loss = loss.sum() / (labels >= 0).sum()
            log_dict["lm_loss"] = loss.item()

        if self.config.use_binary_decision_head:
            w2t_logits_neg = outputs.w2t_logits[neg_frame_mask]
            if len(w2t_logits_neg) != 0:
                labels_neg = torch.ones_like(w2t_logits_neg)
                w2t_loss_neg = bce_loss(w2t_logits_neg, labels_neg)

                w2t_logits_pos = outputs.w2t_logits[pos_frame_mask]
                labels_pos = torch.zeros_like(w2t_logits_pos)
                w2t_loss_pos = bce_loss(w2t_logits_pos, labels_pos)
                # NOTE: the labels are flipped to keep consistent with not using
                # the binary decision head

                w2t_loss = (w2t_loss_neg + w2t_loss_pos) / 2
                log_dict["w2t_loss"] = w2t_loss.item()

                if loss is not None:
                    loss = loss + w2t_loss * self.config.binary_loss_weight
                else:
                    loss = w2t_loss

        if not return_dict:
            if loss is not None:
                if self.training:
                    return (loss,) + outputs[1:] + (log_dict,)
                else:
                    return (loss,) + outputs[1:]
            else:
                return outputs

        outputs.loss = loss
        outputs.log_dict = log_dict
        return outputs


def trim_past_key_values(
    past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor]],
    start: int,
    stop: int,
    batch_idx: int = -1,
) -> KV_CACHE:
    """Select a slice of past key values

    :param past_key_values: Tuple of `tuple(torch.FloatTensor)` of length
        `num_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`).
        This is also known as the legacy cache format of HF.
    :param start: The start index of the slice
    :param stop: The stop index of the slice
    :param batch_idx: The batch index to select the slice from. If -1, select
        the slice for all batches.

    :return: The trimmed past key values
    """
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
