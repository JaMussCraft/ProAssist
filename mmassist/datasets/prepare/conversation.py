import copy
import random
from transformers import AutoTokenizer
from mmassist.datasets.utils.video_utils import time_to_frame_index


def split_conversation(
    conversation: list[dict],
    max_seq_len: int,
    tokenizer: AutoTokenizer,
    keep_ctx_length: tuple[int, int] | None = None,
    fps: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Split a conversation into two parts, where the first part ends with an
    assistant turn before the sequence length exceeds the maximum, and the
    second part starts with the next turn after the assistant turn.

    :param conversation: The conversation to split.
    :param max_seq_len: The maximum sequence length.
    :param tokenizer: The tokenizer for text encoding.
    :param keep_ctx_length: The video context to keep in the first frames
        turn of the second part. A random integer between the two values
        will be chosen as the number of seconds to keep.
    :param fps: The frames per second of the video.

    :return: A tuple containing the splited conversations.
    """
    chat_formatter = tokenizer.chat_formatter
    conversation = copy.deepcopy(conversation)

    seq_length = 0
    seq_length_until_last_assist_turn = 0
    latest_assist_turn_idx = -1
    for turn_idx, turn in enumerate(conversation):
        if turn["role"] != "frames":
            formatted = chat_formatter.add_message(turn)
            is_first_turn = turn_idx == 0
            turn_len = len(
                tokenizer.encode(formatted, add_special_tokens=is_first_turn)
            )
        else:
            num_frames = turn["end"] - turn["start"]
            turn_len = num_frames * chat_formatter.num_tokens_per_img
            if chat_formatter.sep_token:
                turn_len += num_frames - 1

        seq_length += turn_len
        if turn["role"] == "assistant":
            if seq_length > max_seq_len:

                if latest_assist_turn_idx == -1:
                    # this means there is no assistant turn in the current split
                    if turn_idx == 0:
                        # should not happen, but just in case
                        return [], []
                    turn_before = conversation[turn_idx - 1]
                    if turn_before["role"] != "frames":
                        # we should have ensured that there is a "frames" turn before
                        # an assistant turn. If not, discard the current clip
                        return [], []

                    # The frame turn before the assistant turn is too long to fit in.
                    # Since there is no assistant turn in the current split, which means
                    # there is no learning signal in the current split, we can discard
                    # the current clip. And to use the frames later than this frame turn,
                    # we reduce the context for the next clip
                    if keep_ctx_length is not None:
                        end_idx = turn_before["end"]
                        ctx_len = random.randint(*[fps * s for s in keep_ctx_length])
                        new_start_idx = max(0, end_idx - ctx_len)
                        turn_before["start"] = new_start_idx
                    return [], conversation[turn_idx - 1 :]

                first_part = conversation[: latest_assist_turn_idx + 1]
                second_part = conversation[latest_assist_turn_idx + 1 :]

                if not second_part or not any(
                    t["role"] == "assistant" for t in second_part
                ):
                    # print("Seq length:", seq_length_until_last_assist_turn)
                    return first_part, []

                next_turn = second_part[0]
                if next_turn["role"] == "frames":
                    # reduce the video context to keep_ctx_length seconds
                    if keep_ctx_length is not None:
                        end_idx = next_turn["end"]
                        original_num_frames = end_idx - next_turn["start"]
                        ctx_len = random.randint(*[fps * s for s in keep_ctx_length])
                        new_start_idx = max(0, end_idx - ctx_len)
                        second_part[0]["start"] = new_start_idx
                    # print(f"context #frames: {original_num_frames} -> {ctx_len}")

                # print("Seq length:", seq_length_until_last_assist_turn)
                return first_part, second_part

            seq_length_until_last_assist_turn = seq_length
            latest_assist_turn_idx = turn_idx

    # print("Seq length:", seq_length_until_last_assist_turn)
    if latest_assist_turn_idx != -1:
        # drop the non-assisant turns after the last assistant turn
        return conversation[: latest_assist_turn_idx + 1], []
    # no assistant turn in the conversation: discard the whole conversation
    return [], []


def convert_conversation_time_to_index(
    conversation: list[dict],
    fps: float,
    len_frames: int,
    start_time: float = 0.0,
    knowledge: str = "",
) -> list[dict]:
    """Convert the timestamped conversation into the chat format
    of the form: [frames, turn, frames, turn, ...]. Adjust the frame index
    to ensure the frames are in the correct order and not out of range.
    """

    output = []
    is_first_user_turn = True
    start_idx = time_to_frame_index(start_time, fps, "floor")
    for turn in conversation:
        time = turn["time"]
        frame_idx = time_to_frame_index(time, fps, "floor")

        if frame_idx >= len_frames:
            # if frame_idx > len_frames:
            #     print(f"frame idx {frame_idx} exceeds #frames {len_frames}")
            frame_idx = len_frames - 1

        if frame_idx == start_idx:
            # this is normal as we encourage the assistant to talk at the same time of user
            frame_idx += 1
            if frame_idx >= len_frames:
                # print(video_uid)
                # print(f"drop the turn as the frame index out of range: {turn}")
                break
            # if tidx + 1 < len(conversation):
            #     # also increment the time for the next turn
            #     next_turn = conversation[tidx + 1]
            #     next_turn["time"] += 1 / fps
        elif frame_idx < start_idx:
            if start_idx + 1 < len_frames:
                # print(f"Increase {frame_idx} to {start + 1}!")
                frame_idx = start_idx + 1
            else:
                # print(video_uid)
                # print(f"Cannot increase {frame_idx} to {start + 1}!")
                # print(f"drop: {turn}")
                break

        assert frame_idx > start_idx, f"frame_idx: {frame_idx}, start: {start_idx}"
        assert frame_idx < len_frames

        output.append({"role": "frames", "start": start_idx, "end": frame_idx})
        output.append(turn)
        if knowledge and turn["role"] == "user" and is_first_user_turn:
            # add the knowledge of the task after the first user turn
            output.append({"role": "system", "content": knowledge})
            is_first_user_turn = False
        start_idx = frame_idx

    return output
