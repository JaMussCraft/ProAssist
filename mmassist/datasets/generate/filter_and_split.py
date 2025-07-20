import os
import json
import random
import argparse
import math

random.seed(42)

USER_TYPES = ["no_talk", "talk_some", "talk_more"]


def filter_eval_data(data: list[dict], min_score: float) -> list[dict]:
    filtered_data = []
    for sample in data:
        user_type_to_conv = {}
        for conv in sample["conversations"]:
            score = conv["auto_quality_eval"]["final_score"]
            user_type = conv["user_type"]
            if score < min_score:
                continue
            if user_type not in user_type_to_conv:
                user_type_to_conv[user_type] = conv
            else:
                exist_conv = user_type_to_conv[user_type]
                exist_score = exist_conv["auto_quality_eval"]["final_score"]
                if score > exist_score:
                    user_type_to_conv[user_type] = conv

        if len(user_type_to_conv) == len(USER_TYPES):
            sample["conversations"] = [user_type_to_conv[ut] for ut in USER_TYPES]
            filtered_data.append(sample)

    return filtered_data


def filter_train_data(
    data: list[dict], min_score: float, sampling_ratio: float
) -> list[dict]:
    filtered_data = []
    for sample in data:
        user_type_to_convs = {}
        for conv in sample["conversations"]:
            score = conv["auto_quality_eval"]["final_score"]
            user_type = conv["user_type"]
            if score < min_score:
                continue
            if user_type not in user_type_to_convs:
                user_type_to_convs[user_type] = []
            user_type_to_convs[user_type].append(conv)

        kept_convs = []
        for user_type, convs in user_type_to_convs.items():
            convs = sorted(convs, key=lambda x: x["auto_quality_eval"]["final_score"])
            if sampling_ratio > 1:
                sampling_ratio = int(sampling_ratio)
                kept_convs.extend(convs * sampling_ratio)
            else:
                num_convs = math.ceil(len(convs) * sampling_ratio)
                kept_convs.extend(convs[:num_convs])

        if kept_convs:
            sample["conversations"] = kept_convs
            filtered_data.append(sample)

    return filtered_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_filter_score", type=float, default=3)
    parser.add_argument("--sampling_ratio", type=float, default=1)
    parser.add_argument("--eval_filter_score", type=float, default=5)
    args = parser.parse_args()

    subfolder = "generated_dialogs"

    ##### Filter out low-quality dialogs in the training set ###################
    # 1. Rank the dialogs for each user type
    # 2. Filter out the dialogs with the score lower than the threshold
    # 3. (optional) downsample/upsample the data
    ############################################################################

    train_data_file = os.path.join(args.data_dir, subfolder, "train.json")
    print("Train data file:", train_data_file)
    with open(train_data_file, "r") as f:
        train_data = json.load(f)
    num_conv_per_video = len(train_data[0]["conversations"])
    num_conversations = len(train_data) * num_conv_per_video
    print(f"Loaded data of {len(train_data)} videos, {num_conversations} dialogs)")

    # filter the data
    train_data = filter_train_data(
        train_data, args.train_filter_score, args.sampling_ratio
    )
    num_train_convs = sum([len(sample["conversations"]) for sample in train_data])
    print(f"Filtered data: {len(train_data)} videos, {num_train_convs} dialogs")

    save_file = os.path.join(args.data_dir, f"{subfolder}/train_filtered.json")
    with open(save_file, "w") as f:
        json.dump(train_data, f)

    ##### Process the original `val` data to produce our `val` and `test` set ##
    # 1. Get the best dialog for each user type
    # 2. Filter out the dialogs with the score lower than the threshold
    # 3. Keep the videos that have all the 3 user types
    # 4. Split the remaining videos into val and test set with a ratio of 1:1
    ############################################################################
    val_data_file = os.path.join(args.data_dir, subfolder, "val.json")
    print("Val data file:", val_data_file)

    # load the data
    with open(val_data_file, "r") as f:
        val_data = json.load(f)
    num_conv_per_video = len(val_data[0]["conversations"])
    num_conversations = len(val_data) * num_conv_per_video
    print(f"Loaded data of {len(val_data)} videos, {num_conversations} dialogs)")

    # filter the data
    val_data = filter_eval_data(val_data, args.eval_filter_score)
    print(f"Filtered data: {len(val_data)} videos, {len(val_data)*3} dialogs")

    # split the data
    split_file = os.path.join(args.data_dir, "val_test_split.json")
    split_val, split_test = [], []
    if os.path.exists(split_file):
        print(f"Spliting with the existing split file: {split_file}")
        with open(split_file, "r") as f:
            split_info = json.load(f)
        for d in val_data:
            video_uid = d["video_uid"]
            if split_info[video_uid][0] == "val":
                split_val.append(d)
            else:
                split_test.append(d)
    else:
        print("Generating a new split...")
        split_ratio = 0.5
        random.shuffle(val_data)
        split_val = val_data[: int(split_ratio * len(val_data))]
        split_test = val_data[int(split_ratio * len(val_data)) :]
        split_info = {}
        for d in split_val:
            split_info[d["video_uid"]] = ["val", d["inferred_goal"]]
        for d in split_test:
            split_info[d["video_uid"]] = ["test", d["inferred_goal"]]

        # save the split info
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)

    with open(os.path.join(args.data_dir, f"{subfolder}/val_filtered.json"), "w") as f:
        json.dump(split_val, f, indent=2)
    with open(os.path.join(args.data_dir, f"{subfolder}/test_filtered.json"), "w") as f:
        json.dump(split_test, f, indent=2)
