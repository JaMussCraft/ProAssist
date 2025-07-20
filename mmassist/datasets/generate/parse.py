import re
from enum import Enum


class Initiativity(Enum):
    INITIATIVE = "initiative"
    RESPONSIVE = "responsive"

    @staticmethod
    def parse(value: str) -> str:
        value = value.lower()
        if "initiative" in value:
            return Initiativity.INITIATIVE.value
        elif "responsive" in value:
            return Initiativity.RESPONSIVE.value
        raise ValueError(f"Failed to parse initiativity: {value}")


class Intention(Enum):
    INSTRUCTION = "instruction"
    CORRECTION = "correction"
    INFO_SHARING = "info_sharing"
    FEEDBACK = "feedback"
    OTHER = "other"

    @staticmethod
    def parse(value: str) -> str:
        labels = []
        value = value.lower()
        if "instruction" in value:
            labels.append(Intention.INSTRUCTION.value)
        if "correction" in value:
            labels.append(Intention.CORRECTION.value)
        if "info" in value:
            labels.append(Intention.INFO_SHARING.value)
        if "feedback" in value:
            labels.append(Intention.FEEDBACK.value)
        if not labels:
            labels.append(Intention.OTHER.value)
        return ",".join(labels)


def parse_text_to_conversation_dict(text: str, parse_labels: bool = False):
    conversation = []
    lines = text.split("\n")
    for l in lines:
        matched = re.match(r"\[(.*?)s\] (.*?): (.*)", l)
        if not matched:
            continue
        time, role, content = matched.groups()
        # check if time is valid
        try:
            time = float(time)
        except:
            continue

        # check if role is valid
        role = role.lower()
        if role not in ["user", "assistant"]:
            continue

        # check if content is valid
        content = content.strip()
        if content.startswith("("):
            continue

        # extract the labels for the assistant utterance
        if parse_labels:
            labels = ""
            if role == "assistant":
                initia, intent = "UNK", "UNK"
                if "|" in content:
                    labels = re.search(r"\[(.*?)\|(.*?)\]", content)
                    if labels:
                        initia_raw, intent_raw = labels.groups()
                        initia = Initiativity.parse(initia_raw)
                        intent = Intention.parse(intent_raw)
                if initia == "UNK" or intent == "UNK":
                    print(f"Failed to parse labels: {content}")
                    initia, intent = "initiative", "other"
                labels = f"{initia}|{intent}"

        # clean up the content
        content = re.sub(r"\[.*?\|.*?\]", "", content).strip()

        this_turn = {"role": role, "time": time, "content": content}
        if parse_labels:
            this_turn["labels"] = labels
        last_role = conversation[-1]["role"] if conversation else None
        last_time = conversation[-1]["time"] if conversation else None
        if last_role == role and last_time == time:
            if content != conversation[-1]["content"]:
                conversation[-1]["content"] += f" {content}"
                print(f"Merge duplicate into: {conversation[-1]['content']}")
            else:
                print(f"Skip duplicate: {this_turn} vs {conversation[-1]}")
            continue
        conversation.append(this_turn)

    return conversation


def conversation_dict_to_text(
    conversation: list, add_labels: bool = False, max_turns_to_keep: int = -1
):
    text = ""

    drop_turn_idx = -1
    if max_turns_to_keep > 0 and len(conversation) > max_turns_to_keep:
        trunc_len = max_turns_to_keep // 2
        conversation = conversation[:trunc_len] + conversation[-trunc_len:]
        drop_turn_idx = trunc_len - 1

    for idx, c in enumerate(conversation):
        text += f"[{c['time']}s] {c['role'].capitalize()}: {c['content']}"
        if add_labels and c.get("labels"):
            text += f" [{c['labels']}]"
        text += "\n"

        if idx == drop_turn_idx:
            text += "...\n"

    return text
