from dataclasses import dataclass, asdict
from collections import Counter
from llm_utils import LLMGenerator
from parse import conversation_dict_to_text, parse_text_to_conversation_dict
from egoexolearn_tasks import EGOEXOLEARN_TASKS, get_task_descriptions
from wtag_recipes import get_task_and_recipe

# fmt: off
DEFAULT_SYS_PROMPT = "You are a helpful assistant that follows the user's request."


### Prompts for inferring the task goal and instructional knowledge

EGO4D_RECIPE_GEN_PROMPT_TEMPLATE = """Here is a video description of an experienced user working on the task - {goal_description}:
{step_descriptions}

Try to infer the **high-level** recipe from the descriptions. Note that the steps may not belong to the same trial, so you have to infer the correct order of the steps based on common sense, and re-order the steps if necessary. Do not hallucinate details that are not mentioned in the descriptions. Also generate a more **informative** and **descriptive** name for the task based on provided descriptions. The name should be a description of the task, instead of the name of the recipe. 

Give plain and concise text with numbered key steps in the following format:
[task name]
1. ...
2. ...
"""

KNOWLEDGE_GEN_PROMPT_TEMPLATE = """Here is a video description of an user working on {goal_description}:
{step_descriptions}

Try to infer the **high-level** {knowledge_type} from the descriptions. Note that some actions may not irrelevant to the task or have mistakes, so you have to infer the essential and correct steps based on common sense. Do not hallucinate details that are not mentioned in the descriptions. Also generate a more **informative** and **descriptive** name for the task based on provided descriptions. The name should be a description of the task, instead of the name of the {knowledge_type}.

Give plain and concise text with numbered key steps in the following format:
[task name]
1. ...
2. ...
"""

KNOWLEDGE_REFINE_PROMPT_TEMPLATE = """Here are {num_repeats} {knowledge_type}s:
{knowledges}

Some may be incorrect or incomplete. Please give a single correct and complete {knowledge_type} for the task, with numbered key steps. Pick the title that is descriptive for the task, instead of a {knowledge_type} name.

Give plain, unformatted and concise text with numbered key steps in the following format:

[task name]
1. ...
2. ...

Do not include any other information or note."""

KNOWLEDGE_MATCH_PROMPT_TEMPLATE = """Here is a video description of an user working on a task:
{step_descriptions}

The task is from one of the following tasks:
{tasks}

Please select the task that best matches the video description. Give the final answer in the following format:
(whatever thought process you have)
ANSWER: <task id of a single integer>
"""

### Prompts for video categorization

VIDEO_LABEL_PROMPT_TEMPLATE = """Here is a video description of an user working on the task - {goal_description}:
{step_descriptions}

Reference {knowledge_type}:
{knowledge}

Is this a {knowledge_type}? If so, was the user likely to:
1. perform the task roughly following the {knowledge_type} (**no** need to be strict), OR
2. perform other tasks (or another trial of the same task) simultaneously in a multi-tasking manner?

Answer with your analysis, and end your response with "Final answer: 1, 2 or 0" (0 denotes that the activity is not related to {domain})."""

### Prompts for dialog simulation

DIALOG_GEN_SYS_PROMPT = "You are an expert of imagining conversations between users and assistants."

DIALOG_GEN_USER_REQUIREMENTS = {
    "no_talk": "- The user follows the assistant's instructions and does not talk.",
    "talk_some": "- The user asks a few questions or confirm about the instructions, accounting for about 20% of the steps.\n",
    "talk_more": "- The user is talkative and may ask questions that can either be related to the task or not, accounting for about 40% of the steps.\n"
}

DIALOG_GEN_PROMPT_TEMPLATE = """Here is a video description of an user working on the task - {goal_description}:
{step_descriptions}

Your goal is to simulate a conversation between the user and an assistant, where the user's actions are performed following the assistant's instructions. The user will first mention the overall goal of the task. The assistant informs the user about the next step at proper time. Importantly, the assistant is proactive and always provides the next step even before the user asks for it. Before the task starts, the assistant may also give a brief introduction about the task. {additional_requirement}

Requirements for the assistant:
- Time is crucial! Try to generate the dialog that strictly aligns with the video timeline.
- Try to cover all the essential steps in the task. If the user asks a question at the time the assistant should give the next step, the assistant turn should include both the response to the question and instruction about the next step.
- Be helpful and friendly. If the user asks something that has been explained before, the assistant should still provide the information with patience.
- Try to be encouraging when the user makes progress, but do not overdo it.
- Be concise! The dialog is verbal, so avoid long sentences.
- Do not say "can you do it for me" to the user.


Requirements for the user:
{user_requirement}


Generation format:
[time] User: ...
[time] Assistant: ...
[time] Assistant: ...
[time] User: ...
[time] Assistant: ...

Note that the minimal interval between each turn is 1 second, which means the user will wait for at least 1 second after an assistant's turn, and two consecutive assistant's turns should have at least 1 second interval. Combine close turns into a single turn if necessary. One exception is that the assistant must respond **immediately** when the user says something (i.e. give a response right after an user's turn at the same time).

{dialog_history}

In this round, please **only** generate the dialog for the video from time [{start_time:.1f}s] to [{end_time:.1f}s]!"""

ADDITIONAL_REQUIREMENTS = {
    # "holoassist": "Note that the video description contains both the user's actions and the user-assistant dialog. Anchor the dialog to the **key steps** of the task, not every single action of the user. Errors made by the user and the timing of original dialog can be a strong hint for when to simulate the dialog. You may rephrase the dialog to make it more coherent and human-like.", 
    "holoassist": "Note that the video description contains both the user's actions and the user-assistant dialog. Anchor the simulated dialog to the existing dialog, and try to rephrase the utterances to make them more coherent and human-like. You may add a few more turns around the **essential steps** of the task, which are the underlying intentions of the action instead of the actions themselves. Add a few turns to make the dialog more fluent and helpful, but avoid being overwhelming.",
    "egoexolearn": "The simulated dialog should be centered around the **key steps** of the task, not every single action of the user. Try to make the dialog more coherent and helpful as what a human assistant will say.",
    "epickitchens": "The simulated dialog should be centered around the **key steps** of the task, not every single action of the user. Note that the user may make mistake or perform suboptimal actions, the assistant should not give instructions on those actions, but smartly select right time to give guidance. Try to make the dialog more coherent and helpful as what a human assistant will say.",
    "wtag": "Note that the video description contains both the step description and the user-assistant dialog. Anchor the simulated dialog to the existing dialog, and try to rephrase the utterances to make them more coherent and human-like. Add more details such as assistant feedback or user question during long steps if necessary. Remember to generate the response to user's question even if there isn't one in the original dialog from the video description.", 
    "assembly101": "\n\nThe mistakes made by the user are marked by (mistake: <mistake type>). If a mistake happens, we want to simulate the dialog in the way that the assistant helps the user correct the mistake. To be more specific, the assistant SHOULD NOT give instructions if an action is 'wrong order', 'previous one is mistake' or 'shouldn't have happened'. Instead, the assistant should give instruction of the CORRECT next step (i.e. scan the future actions and select the nearest correct action). Afterwards, at the start of actions marked as 'correction', the assistant should mention the previous mistake and give insruction on how to correct it based on the corrective action. For 'wrong position' mistakes, the assistant can give the instruction of that action, but need to point out the mistake at the start time of corrective action for that mistake.",
}


DIALOG_REFINE_AND_LABEL_PROMPT_TEMPLATE = """Here is a conversation between a user and an assistant:
{dialog_history}

For each assistant message, add labels regarding the assistant's initiativity and intention:

Initiativity:
- initiative: The assistant says something proactively without the user asking for it.
- responsive: The assistant responds to the user's question or comment.

Intention:
- instruction: The assistant gives an instruction to the user.
- correction: The assistant corrects a mistake made by the user, either proactively or responsively. Suggestions for alternative actions can also be included.
- info_sharing: The assistant shares some information with the user, such as explaining something or giving a tip.
- feedback: The assistant gives feedback to the user, such as "good job" or "tips for improvement".
- other: Other intentions that do not fall into the above categories.

Intention can be multiple, e.g., "instruction, info_sharing".


Generation format:
[time] User: ...
[time] Assistant: ... [initiativity|intentions]
[time] Assistant: ... [initiativity|intentions]
[time] User: ...
[time] Assistant: ... [initiativity|intentions]

When generating the dialog, you should also refine the dialogue following these guidelines:
1. Merge turns that are close in time (less than 1 second apart) into a single turn, when the content is similar or related.
2. Use more coreference and pronouns to make the dialog more coherent and human-like.
3. Decide the length of assistant messages smartly. Make them more clear and helpful when necessary, but keep them concise and to the point in general.
4. Avoid repeating the same talking patterns or phrases. For example, do not say "make sure ..." for every instruction.
5. Rephrase impolite or inappropriate language, such as "as I have mentioned this earlier ...", to be more friendly and helpful. But keep concise and to the point.
6. Remove anything other than the dialog itself, such as the user's actions or explanations of how the dialog is generated.
Do not just copy paste the original dialog!"""


SUMMARY_SYS_PROMPT = "You are an expert of summarizing conversations."

PROGRESS_SUMMARY_PROMPT_TEMPLATE = """Here is a conversation between a user and an assistant:
{dialog_history}

Summarize the task goal and progress so far, including:
1. The task goal mentioned by the user.
2. What has been done.
3. Other topics mentioned by the user in the conversation, if any.
4. The current state/step of the task.
Be faithful and try to include all the relevant information.

Give your response in plain text of a single line in the following format:
SUMMARY: <progress summary>
"""
# fmt: on


def retry_on_failure(max_repeats: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_repeats):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Failed with error: {e}. Retry {i+1}/{max_repeats}")
            raise Exception(f"Failed after {max_repeats} retries")

        return wrapper

    return decorator


@retry_on_failure()
def infer_goal_and_knowledge(
    dataset_name: str,
    goal_description: str,
    step_descriptions: str,
    knowledge_type: str,
    llm: LLMGenerator,
    num_repeats: int = 10,
) -> tuple[str, str]:

    # generate {num_repeats} pieces of knowledge based on the video descriptions
    if dataset_name == "egoexolearn":
        # use LLM to select from GT tasks & recipes for egoexolearn
        tasks, task_descs = EGOEXOLEARN_TASKS, get_task_descriptions()
        return match_task(step_descriptions, llm, tasks, task_descs, num_repeats)
    elif dataset_name == "wtag":
        # can simply get the GT recipe by key word matching for WTaG
        return get_task_and_recipe(step_descriptions)
    elif dataset_name == "ego4d":
        gen_prompt = EGO4D_RECIPE_GEN_PROMPT_TEMPLATE.format(
            goal_description=goal_description, step_descriptions=step_descriptions
        )
    else:
        goal = "a task" if not goal_description else f"the task - {goal_description}"
        gen_prompt = KNOWLEDGE_GEN_PROMPT_TEMPLATE.format(
            goal_description=goal,
            step_descriptions=step_descriptions,
            knowledge_type=knowledge_type,
        )
    inputs = [("system", DEFAULT_SYS_PROMPT), ("user", gen_prompt)]
    outputs = llm.generate(inputs, n=num_repeats)

    knowledges = ""
    for i, t in enumerate(outputs):
        knowledges += f"{knowledge_type.capitalize()} {i+1}:\n {t}\n\n"

    # refine the knowledges into a single correct and complete manual
    refine_prompt = KNOWLEDGE_REFINE_PROMPT_TEMPLATE.format(
        num_repeats=num_repeats,
        goal_description=goal_description,
        knowledge_type=knowledge_type,
        knowledges=knowledges,
    )
    inputs = [("system", DEFAULT_SYS_PROMPT), ("user", refine_prompt)]
    refined_knowledge = llm.generate(inputs)[0]

    # parse the inferred goal
    inferred_goal = refined_knowledge.split("\n")[0].replace("*", "").strip()
    return inferred_goal, refined_knowledge


@retry_on_failure()
def match_task(
    step_descriptions: str,
    llm: LLMGenerator,
    tasks: list[dict],
    task_descs: str,
    num_repeats: int = 10,
) -> tuple[str, str]:

    # generate {num_repeats} pieces of knowledge based on the video descriptions
    prompt = KNOWLEDGE_MATCH_PROMPT_TEMPLATE.format(
        step_descriptions=step_descriptions, tasks=task_descs
    )
    inputs = [("system", DEFAULT_SYS_PROMPT), ("user", prompt)]
    outputs = llm.generate(inputs, n=num_repeats)

    # parse and count
    answers = Counter()
    for o in outputs:
        ans = o.lower().split("answer:")[1]
        import re

        matched = re.search(r"\d+", ans)
        if matched:
            task_id = int(matched.group())
        else:
            continue
        answers[task_id] += 1

    task_id = answers.most_common(1)[0][0] - 1

    goal = tasks[task_id]["name"]
    knowledge = goal + "\n"
    for s_idx, step in enumerate(tasks[task_id]["steps"]):
        knowledge += f"{s_idx + 1}. {step}\n"
    return goal, knowledge


@retry_on_failure()
def label_video(
    goal_description: str,
    step_descriptions: str,
    knowledge: str,
    knowledge_type: str,
    domain: str,
    llm: LLMGenerator,
    num_repeats: int = 10,
) -> Counter:
    prompt = VIDEO_LABEL_PROMPT_TEMPLATE.format(
        goal_description=goal_description,
        step_descriptions=step_descriptions,
        knowledge=knowledge,
        knowledge_type=knowledge_type,
        domain=domain,
    )
    inputs = [("system", DEFAULT_SYS_PROMPT), ("user", prompt)]

    # generate {num_repeats} labels for the video
    outputs = llm.generate(inputs, n=num_repeats)

    # count the number of labels
    answers = Counter()
    for o in outputs:
        parsed_ans = o.lower().split("answer: ")[1]
        label = 1 if "1" in parsed_ans else 2 if "2" in parsed_ans else 0
        answers[label] += 1

    return answers


def adjust_time(conversation: list[dict], time_shift: float = 1.0) -> list[dict]:
    """Adjust the time of the turns in the conversation to ensure the minimal
    interval between each turn is larger than "time_shift" seconds."""
    while True:
        adjusted = False
        for idx, turn in enumerate(conversation):
            if idx == 0:
                continue

            last_turn = conversation[idx - 1]
            if turn["time"] - last_turn["time"] < 0.5 and (
                (turn["role"], last_turn["role"]) != ("assistant", "user")
            ):
                adjust_turns = [turn]
                if idx + 1 < len(conversation):
                    next_turn_idx = idx + 1
                    next_turn = conversation[next_turn_idx]
                    while (
                        next_turn["time"] - turn["time"] < time_shift
                        and next_turn_idx < len(conversation) - 1
                    ):
                        adjust_turns.append(next_turn)
                        next_turn_idx += 1
                        next_turn = conversation[next_turn_idx]

                for at in adjust_turns:
                    at["time"] = last_turn["time"] + time_shift
                    adjusted = True

        if not adjusted:
            break

    return conversation


@retry_on_failure()
def generate_conversation(
    goal_description: str,
    clips: list[tuple[float, float, str]],
    llm: LLMGenerator,
    user_types: list[str],
    additional_requirement: str = "",
) -> list[str]:

    user_reqs = [DIALOG_GEN_USER_REQUIREMENTS[p] for p in user_types]

    batch_conv = [[] for _ in range(len(user_reqs))]
    for st, et, desc in clips:
        print(f"Generating dialog for {st:.1f}s-{et:.1f}s")
        batch_inputs = []
        for i, user_req in enumerate(user_reqs):
            dialog_history = conversation_dict_to_text(
                batch_conv[i], add_labels=True, max_turns_to_keep=20
            )
            if dialog_history:
                dialog_history = f"You have already generated the following dialog:\n{dialog_history}"
            prompt = DIALOG_GEN_PROMPT_TEMPLATE.format(
                goal_description=goal_description,
                step_descriptions=desc,
                user_requirement=user_req,
                dialog_history=dialog_history,
                start_time=st,
                end_time=et,
                additional_requirement=additional_requirement,
            )
            # print("dialog generation", prompt)
            batch_inputs.append([("system", DIALOG_GEN_SYS_PROMPT), ("user", prompt)])

        # parallel generate for all user profiles
        outputs = llm.batch_generate(batch_inputs)

        # add the generated dialog to the conversation history
        clip_convs = []
        for output in outputs:
            conv_dict = parse_text_to_conversation_dict(output[0])
            conv_dict = [c for c in conv_dict if c["time"] <= et]
            clip_convs.append(conv_dict)
            # print(inputs[1][1])
            # print(f"Conversation {idx}, before refinement")
            # print(conversation_dict_to_text(conv_dict, add_labels=True))

        clip_convs_refined = refine_and_label_dialog(clip_convs, llm)
        for idx, conv in enumerate(clip_convs_refined):
            batch_conv[idx].extend(conv)

    # refine the dialogs and add assistant intention labels
    # batch_conv = refine_and_label_dialog(batch_conv, llm)
    # for idx, conv in enumerate(batch_conv):
    #     print(f"Conversation {idx}, after refinement")
    #     print(conversation_dict_to_text(conv, add_labels=True))

    conv_with_user_type = [
        {"conversation": c, "user_type": p} for c, p in zip(batch_conv, user_types)
    ]
    return conv_with_user_type


@retry_on_failure()
def refine_and_label_dialog(conversations: list[list[dict]], llm: LLMGenerator) -> dict:
    batch_inputs = []
    for conv in conversations:
        dh = conversation_dict_to_text(conv)
        # print("before refine", dh)
        prompt = DIALOG_REFINE_AND_LABEL_PROMPT_TEMPLATE.format(dialog_history=dh)
        inputs = [("system", SUMMARY_SYS_PROMPT), ("user", prompt)]
        batch_inputs.append(inputs)

    # generate the refined dialogs in batch
    batch_outputs = llm.batch_generate(batch_inputs)

    # update the conversations with the refined dialogs
    refined_conversations = []
    for outputs in batch_outputs:
        # print("after refine", outputs[0])
        conv = parse_text_to_conversation_dict(outputs[0], parse_labels=True)
        refined_conversations.append(conv)

    return refined_conversations


@retry_on_failure()
def add_progress_summary(conversation: list[dict], llm: LLMGenerator) -> dict:
    batch_inputs = []
    summ_turn_ids = []
    for idx, turn in enumerate(conversation):
        if turn["role"] == "assistant":
            dh = conversation_dict_to_text(conversation[: idx + 1], add_labels=False)
            prompt = PROGRESS_SUMMARY_PROMPT_TEMPLATE.format(dialog_history=dh)
            inputs = [("system", SUMMARY_SYS_PROMPT), ("user", prompt)]
            batch_inputs.append(inputs)
            summ_turn_ids.append(idx)

    # generate the progress summary in batch
    batch_outputs = llm.batch_generate(batch_inputs)

    # update the conversation with the progress summary
    for turn_idx, outputs in zip(summ_turn_ids, batch_outputs):
        time = conversation[turn_idx]["time"]
        elsp = f"The time elapsed since the start of the task is {time:.1f} seconds. "

        progress = None
        for l in outputs[0].split("\n"):
            if "SUMMARY" in l:
                progress = elsp + l.split(":")[1].strip()
        if progress is None:
            raise ValueError(f"Failed to parse: {outputs[0]}")

        conversation[turn_idx]["progress"] = progress

    return conversation


@dataclass
class ParsedVideoAnns:
    dataset: str
    domain: str  # "cooking", "object manipulation", "lab"
    knowedge_type: str  # "cooking recipe", ...
    video_uid: str
    goal_description: str
    all_step_descriptions: str
    clips: list[tuple[float, float, str]]
    duration: float
    ann_ratio: float
    num_steps: int
    video_start_time: float = 0.0
    has_mistake: bool = False
    num_substeps: int | None = None
    fps: float | None = None
    original_ann: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratedOutputs:
    video_uid: str
    inferred_goal: str
    inferred_knowledge: str
    video_labels: Counter
    conversations: list[dict[str, str | list[dict]]]
    parsed_video_anns: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def generate_from_annotation(
    annotation: ParsedVideoAnns,
    llm: LLMGenerator,
    user_types: list[str],
    num_repeats: int = 10,
    use_inferred_goal: bool = True,
    keep_original_anns: bool = True,
    min_ann_ratio: float = 0.5,
    filter_by_llm: bool = False,
) -> GeneratedOutputs | str:

    video_uid = annotation.video_uid
    print(f"Processing video {video_uid}")

    dataset = annotation.dataset
    knowledge_type = annotation.knowedge_type
    goal_description = annotation.goal_description
    step_descriptions = annotation.all_step_descriptions
    ann_ratio = annotation.ann_ratio
    if ann_ratio < min_ann_ratio:
        skip_msg = f"Skip video {video_uid} with low annotation ratio: {ann_ratio}"
        print(skip_msg)
        return skip_msg

    print(
        (
            f"Goal: {goal_description}| Duration: {annotation.duration:.1f}s| "
            f"Num steps: {annotation.num_steps}| Num substeps: {annotation.num_substeps}| "
            f"Num clips: {len(annotation.clips)} | Ann ratio: {annotation.ann_ratio:.2f}"
        )
    )

    # 1. infer goal and recipe
    print("Infer goal and knowledge")
    inferred_goal, inferred_knowledge = infer_goal_and_knowledge(
        dataset, goal_description, step_descriptions, knowledge_type, llm, num_repeats
    )
    print(f"inferred_goal: {inferred_goal}")
    # print(f"inferred_knowledge: {inferred_knowledge}")

    if use_inferred_goal:
        goal_description = inferred_goal

    # 2. label video and filter out inappropriate videos
    if filter_by_llm:
        video_labels = label_video(
            goal_description,
            step_descriptions,
            inferred_knowledge,
            knowledge_type,
            domain=annotation.domain,
            llm=llm,
            num_repeats=num_repeats,
        )
        print("Label video", video_labels)
        label, cnt = video_labels.most_common(1)[0]
        if label != 1 or cnt < num_repeats // 2:
            skip_msg = f"Skip video {video_uid} with label: {video_labels}"
            print(skip_msg)
            return skip_msg
        video_labels = dict(video_labels)
    else:
        video_labels = {}

    # 3. generate the user-assistant conversations
    print("Generate conversations")
    clips = annotation.clips
    add_reqs = ADDITIONAL_REQUIREMENTS.get(annotation.dataset, "")
    if dataset == "assembly101" and not annotation.has_mistake:
        add_reqs = ""
    conversations = generate_conversation(
        goal_description, clips, llm, user_types, add_reqs
    )
    # adjust time
    for conv in conversations:
        conv["conversation"] = adjust_time(conv["conversation"])

    # 4. add progress summary
    print("Add progress summary")
    for conv in conversations:
        conv["conversation"] = add_progress_summary(conv["conversation"], llm)

    # return the generated outputs
    outputs = GeneratedOutputs(
        video_uid=video_uid,
        inferred_goal=inferred_goal,
        inferred_knowledge=inferred_knowledge,
        video_labels=video_labels,
        conversations=conversations,
        parsed_video_anns=annotation.to_dict() if keep_original_anns else None,
    )
    return outputs
