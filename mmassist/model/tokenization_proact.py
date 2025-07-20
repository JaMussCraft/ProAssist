import torch
from typing import Literal
from abc import abstractmethod
from transformers import AutoTokenizer

from mmassist.model.configuration_proact import ProActConfig

Roles = Literal["system", "user", "assistant", "frames"]


def get_learn_ranges_for_img_tokens(
    img_token: str,
    num_tokens_per_img: int,
    num_frames: int,
    img_sep_token: str,
    sampling_rate: float = 1.0,
) -> torch.Tensor:
    """Get the learn ranges for image tokens.

    For example, if the image token is `<i>`, the number of tokens per image is 3,
    and the number of frames is 3, the img_sep_token is `<sep>`, then the image
    token sequence and the learnable positions will be:
    <i><i><i><sep><i><i><i><sep><i><i><i>
    ---------xxxxx---------xxxxx---------

    The learn ranges will be:
    [[9, 14], [23, 28]]

    If sampling_rate < 1.0, then we randomly sample floor(sampling_rate * num_frames)
    frames to learn from. This is useful for dealing with the highly imbalanced
    talk vs no-talk frames in the dataset.
    """
    len_img_tokens_with_sep = len(num_tokens_per_img * img_token + img_sep_token)

    learn_range_start_idxs = torch.arange(
        len_img_tokens_with_sep,
        len_img_tokens_with_sep * num_frames,
        len_img_tokens_with_sep,
    ) - len(img_sep_token)

    len_learn = len(img_sep_token) if img_sep_token else len(img_token)
    learn_ranges = torch.stack(
        [learn_range_start_idxs, learn_range_start_idxs + len_learn], dim=1
    )
    if sampling_rate < 1.0 and num_frames > 1:
        num_learn_frames = max(int(sampling_rate * (num_frames - 1)), 1)
        indexes = torch.randperm(num_frames - 1)[:num_learn_frames]
        learn_ranges = learn_ranges[indexes.sort().values]

    return learn_ranges


class MultimodalChat:
    def __init__(self, img_token: str, num_tokens_per_img: int, sep_token: str) -> None:
        """Initialize the multimodal chat formatter.

        :param img_token: the placeholder token for image features
        :param num_tokens_per_img: the number of tokens per image
        :param sep_token: the separator token for multimodal chat
        """
        # new tokens for multimodal live dialog
        self.img_token = img_token
        self.num_tokens_per_img = num_tokens_per_img
        self.sep_token = sep_token

    @classmethod
    def from_config(cls, config: ProActConfig) -> "MultimodalChat":
        sep_token = config.img_sep_token
        return cls(
            img_token=config.img_token,
            num_tokens_per_img=config.num_tokens_per_img,
            sep_token=sep_token,
        )

    def add_img_tokens(self, num_imgs: int) -> str:
        """Add image tokens to the chat."""
        img_tokens = self.img_token * self.num_tokens_per_img
        return f"{self.sep_token}".join([img_tokens] * num_imgs)

    def add_bos(self) -> str:
        """Add a beginning of the text token."""
        return ""

    @abstractmethod
    def add_message(self, message: dict) -> str:
        """Add a message to the chat.

        :param message: a dict with the following keys
            'role': the role of the message, one of those in `Roles`
            'content' (optional): the content of the text messages
            'start' (optional): the start index of the frames
            'end' (optional): the end index of the frames
        """
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(self, dialog: list[dict]) -> str:
        """Apply the chat template to the dialog."""
        raise NotImplementedError

    @abstractmethod
    def cleanup_text(self, text: str) -> str:
        """Clean up the text."""
        raise NotImplementedError

    def get_learn_ranges(
        self, conversation: list[dict], sampling_rate: float = 1.0
    ) -> list[range]:
        """Get the learn ranges for the conversation.

        Learnable tokens should be are all the tokens from the assistant turn,
        and the last image token in the sequence of tokens for each frame.
        Learnable ranges are returned in the format of a list of ranges of
        char positions in the input string before tokenization. This can be
        assosiated with the token positions with the help of the offset mapping
        returned by the tokenizer.

        :param conversation: the conversation in the format of a list of dicts
        :param chat_formatter: the chat formatter to convert each message in the
            conversation to text string ready to be encoded
        :sampling_rate: sampling rate of negative (i.e. not talk) frames to learn
            from. 0 means not to learn when to talk. A value <1 can help alleviate
            the imbalance between talk and no-talk frames in the dataset.
        :return: a list of ranges for learnable char positions in the input
            string before tokenization
        """
        offset = len(self.add_bos())
        learn_ranges = []
        for turn in conversation:
            input_str = self.add_message(turn)
            if sampling_rate >= 0 and turn["role"] == "frames":
                learn_ranges_img = get_learn_ranges_for_img_tokens(
                    self.img_token,
                    self.num_tokens_per_img,
                    turn["end"] - turn["start"],
                    self.sep_token,
                    sampling_rate,
                )
                learn_ranges_img = learn_ranges_img + offset
                learn_ranges.extend([range(r[0], r[1]) for r in learn_ranges_img])
            elif turn["role"] == "assistant":
                learn_range = range(offset, offset + len(input_str))
                learn_ranges.append(learn_range)
            offset += len(input_str)
        return learn_ranges


class LLaMA3MultimodalChat(MultimodalChat):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # special tokens used chat format in LLaMA-3
        self.bot = "<|begin_of_text|>"
        self.bor = "<|start_header_id|>"
        self.eor = "<|end_header_id|>\n\n"
        self.eot = "<|eot_id|>"

    def add_bos(self) -> str:
        return self.bot

    def add_header(self, message: dict) -> str:
        return f"{self.bor}{message['role']}{self.eor}"

    def add_message(self, message: dict) -> str:
        if message["role"] == "frames":
            return self.add_img_tokens(message["end"] - message["start"])
        return self.add_header(message) + message["content"].strip() + self.eot

    def apply_chat_template(self, dialog: list[dict]) -> str:
        out = self.add_bos()
        for msg in dialog:
            out += self.add_message(msg)
        return out

    def cleanup_text(self, text: str) -> tuple[str, str | None]:
        if self.eot in text:
            # remove the part after EOS, if any
            text = text.split(self.eot, 1)[0]
        # remove the bot token
        text = text.strip().replace(self.bot, "")
        # remove the image tokens
        text = text.replace(self.img_token, "").replace(self.sep_token, "")
        if self.eor in text:
            # extract the role and the text
            role, text = text.split(self.eor, 1)
            role = role.replace(self.bor, "")
            return text, role
        if text == ".":
            text = ""
        return text, None


def build_tokenizer_and_update_config(model_config: ProActConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_pretrained,
        use_fast=True,
        padding_side=model_config.padding_side,
    )

    # update the tokenizer
    vocab = tokenizer.get_vocab()
    new_tokens = []
    if model_config.img_token not in vocab:
        new_tokens.append(model_config.img_token)
    if model_config.img_sep_token and model_config.img_sep_token not in vocab:
        new_tokens.append(model_config.img_sep_token)
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    assert tokenizer.eos_token is not None, "Tokenizer must have an eos token."

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.img_token = model_config.img_token
    tokenizer.img_token_id = tokenizer.convert_tokens_to_ids(tokenizer.img_token)
    tokenizer.img_sep_token = model_config.img_sep_token
    tokenizer.img_sep_token_id = (
        tokenizer.convert_tokens_to_ids(model_config.img_sep_token)
        if model_config.img_sep_token
        else None
    )
    tokenizer.ignore_id = model_config.ignore_id

    # add the chat formatter
    chat_formatter_cls = eval(model_config.chat_formatter_cls)
    chat_formatter = chat_formatter_cls.from_config(model_config)
    tokenizer.chat_formatter = chat_formatter
    tokenizer.bor_token_id = tokenizer.convert_tokens_to_ids(chat_formatter.bor)

    # update the model config
    model_config.img_token_id = tokenizer.img_token_id
    model_config.img_sep_token_id = tokenizer.img_sep_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    model_config.bor_token_id = tokenizer.bor_token_id

    tokenizer.w2t_target_id = model_config.w2t_target_id

    return tokenizer


if __name__ == "__main__":
    config = ProActConfig(
        llm_pretrained="meta-llama/Meta-Llama-3.1-8B-Instruct",
        img_token="<x>",
        img_patch_token_size=3,
        img_sep_token=";",
    )
    tokenizer = build_tokenizer_and_update_config(config)

    system_prompt = "You are a chatbot."
    user = "I, user"
    asst = "I am the assistant"
    asst2 = "I am"
    dialog = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "frames", "start": 0, "end": 3},  # 3 frames
        {"role": "assistant", "content": asst},
        {"role": "frames", "start": 3, "end": 4},  # 1 frame
        {"role": "assistant", "content": asst2},
        {"role": "frames", "start": 4, "end": 9},  # 5 frames
        {"role": "assistant", "content": asst},
    ]
    prompt = tokenizer.chat_formatter.apply_chat_template(dialog)
    print(prompt)

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(input_ids)

    sampling_rate = 0.5
    learn_ranges = tokenizer.chat_formatter.get_learn_ranges(dialog, sampling_rate)
    batch = tokenizer(
        [prompt],
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    batch_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        [prompt], batch_labels, batch.input_ids, batch.offset_mapping, [learn_ranges]
    ):
        for learn_r in learn_range:
            print(learn_r)
            start = torch.nonzero(offset_mapping[:, 0] == learn_r.start).item()
            if offset_mapping[:, 0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:, 0] == learn_r.stop).item()
            else:  # the last eos token
                stop = len(input_ids)
            labels[start - 1 : stop - 1] = input_ids[start:stop]

    if tokenizer.img_sep_token_id is None:
        batch_labels[batch_labels == tokenizer.img_token_id] = tokenizer.eos_token_id

    img_tokens_mask = batch.input_ids == tokenizer.img_token_id
    neg_frame_mask = (batch_labels == config.w2t_target_id) & img_tokens_mask
    pos_frame_mask = (batch_labels == tokenizer.bor_token_id) & img_tokens_mask

    print(prompt)
    print(batch.input_ids)
    print(batch_labels)
