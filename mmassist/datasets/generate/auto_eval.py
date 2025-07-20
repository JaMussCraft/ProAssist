import re
import numpy as np


def round_time(t: float) -> float:
    return float(f"{t:.1f}")


def get_min_time_diff(t: float, ref_times: np.ndarray) -> float:
    """Return the minimum time difference between a time `t` and a
    set of reference times."""
    time_diff = np.abs(ref_times - t)
    return np.min(time_diff)


def auto_eval_generated_conversations(
    generated_dialog: dict, no_resp_penalty_factor: float = 0.1
) -> dict:

    parsed_video_anns = generated_dialog["parsed_video_anns"]
    dataset_name = parsed_video_anns["dataset"]
    step_descriptions = parsed_video_anns["all_step_descriptions"]

    evaluated_conversations = []
    for conversation_dict in generated_dialog["conversations"]:

        start_times, all_times = [], []
        if dataset_name == "ego4d-goalstep":
            # backward compatibility
            for step in parsed_video_anns["original_ann"]["segments"]:
                relevance = step.get("is_relevant", "unk")
                if relevance != "essential":
                    continue
                start_times.append(round_time(step["start_time"]))
                all_times.append(round_time(step["start_time"]))
                all_times.append(round_time(step["end_time"]))
                for sstep in step["segments"]:
                    ss_relevance = step.get("is_relevant", "unk")
                    if ss_relevance != "essential":
                        continue
                    start_times.append(round_time(sstep["start_time"]))
                    all_times.append(round_time(sstep["start_time"]))
                    all_times.append(round_time(sstep["end_time"]))
        else:
            times = re.findall(r"\[(.*?)\]", step_descriptions)
            for t in times:
                if "-" in t:
                    st, et = t.split("-")
                    try:
                        st, et = float(st[:-1]), float(et[:-1])
                        all_times.append(st)
                        all_times.append(et)
                        start_times.append(st)
                    except:
                        pass
                else:
                    try:
                        st = float(t[:-1])
                        all_times.append(st)
                        start_times.append(st)
                    except:
                        pass
        start_times = np.array(start_times)
        all_times = np.array(all_times)

        conversation = conversation_dict["conversation"]
        if not conversation:
            continue
        no_resp_user_turns = 0
        time_diff_p = 0
        talk_times = []
        for idx, turn in enumerate(conversation):
            talk_times.append(turn["time"])
            time_diff_p += get_min_time_diff(turn["time"], all_times)
            if turn["role"] == "user" and turn["content"].endswith("?"):
                if idx == len(conversation) - 1:
                    continue  # no need to count the last turn
                next_turn = conversation[idx + 1]
                if (
                    next_turn["role"] != "assistant"
                    or next_turn["time"] - turn["time"] > 2
                ):
                    # print(turn)
                    no_resp_user_turns += 1

        time_diff_r = 0
        talk_times = np.array(talk_times)
        for t in start_times:
            time_diff_r += get_min_time_diff(t, talk_times)

        # average time difference of talking times against their closest action times
        avg_time_diff_p = time_diff_p / len(conversation)
        # average time difference of action times against their closest talking times
        avg_time_diff_r = time_diff_r / len(start_times)

        time_diff = avg_time_diff_p + avg_time_diff_r
        no_resp_penalty = no_resp_user_turns * no_resp_penalty_factor

        final_score = time_diff + no_resp_penalty  # the lower the better
        final_score = 10 - final_score

        conversation_dict["auto_quality_eval"] = {
            "avg_time_diff_p": avg_time_diff_p,
            "avg_time_diff_r": avg_time_diff_r,
            "no_resp_penalty": no_resp_penalty,
            "no_resp_user_turns": no_resp_user_turns,
            "final_score": final_score,
        }
        evaluated_conversations.append(conversation_dict)

    generated_dialog["conversations"] = evaluated_conversations
    return generated_dialog


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Generated dialogs file.")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        generated_dialogs = json.load(f)

    print(f"Evaluating {len(generated_dialogs)} generated dialogs from {args.file}")
    dialogs_with_eval = []
    for dialog in generated_dialogs:
        try:
            dialog = auto_eval_generated_conversations(dialog)
        except:
            print(f"Failed to evaluate dialog: {dialog}")
            continue
        if dialog["conversations"]:
            dialogs_with_eval.append(dialog)
        else:
            uid = dialog["parsed_video_anns"]["video_uid"]
            print(f"Discard sample with no conversations: {uid}")

    print(f"Saving...")
    with open(args.file, "w") as f:
        json.dump(dialogs_with_eval, f, indent=2)
