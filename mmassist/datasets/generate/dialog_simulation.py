import os
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional, Union, List, Dict, Tuple
from mmassist.datasets.generate.llm_utils import LLMGenerator
from mmassist.datasets.generate.parse import conversation_dict_to_text, parse_text_to_conversation_dict
from mmassist.datasets.generate.egoexolearn_tasks import EGOEXOLEARN_TASKS, get_task_descriptions
from mmassist.datasets.generate.wtag_recipes import get_task_and_recipe

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
    "talk_some": (
        "- The user is moderately engaged and speaks up regularly throughout the task.\n"
        "- Aim for approximately 20-30% of all conversational turns to be from the user (if there are 10 total turns, 2-3 should be user turns).\n"
        "- User behaviors include:\n"
        "  * Asking clarifying questions about instructions (e.g., 'How long should I cook this?', 'Which bowl should I use?')\n"
        "  * Confirming understanding before proceeding (e.g., 'So I mix these together first?', 'Should I turn up the heat?')\n"
        "  * Expressing uncertainty or hesitation (e.g., 'I'm not sure if this is right', 'Does this look okay?')\n"
        "  * Asking visual-based questions about what they observe (e.g., 'Is this the right color?', 'Does this look done?', 'Should it be bubbling like this?')\n"
    ),
    "talk_more": (
        "- The user is highly engaged, talkative, and interactive throughout the entire task.\n"
        "- Aim for approximately 40-50% of all conversational turns to be from the user (if there are 10 total turns, 4-5 should be user turns).\n"
        "- User behaviors include:\n"
        "  * Frequently asking clarifying questions (e.g., 'Should the heat be medium or high?', 'How finely should I chop this?')\n"
        "  * Regularly confirming understanding (e.g., 'Just to confirm, I stir clockwise?', 'You mean the red pepper, right?')\n"
        "  * Providing frequent status updates (e.g., 'I'm mixing now', 'Almost done with this step', 'This is taking longer than expected')\n"
        "  * Asking follow-up questions (e.g., 'Why do we do it this way?', 'Can I use a different ingredient?')\n"
        "  * Expressing observations and concerns (e.g., 'This seems thick', 'I smell something burning', 'The color looks different')\n"
        "  * Making small talk related OR unrelated to the task (e.g., 'I've never made this before', 'My friend loves this dish', 'It's hot in here')\n"
        "  * Asking proactive questions about future steps (e.g., 'What comes after this?', 'Do I need to prepare anything else?')\n"
        "  * Expressing emotions and reactions (e.g., 'This is harder than I thought', 'Wow, that smells great!', 'Oops!')\n"
        "  * Asking visual-based questions about what they see (e.g., 'Is this brown enough?', 'Does this look like the right texture?', 'Is it supposed to look like this?', 'Should there be this much steam?')\n"
    )
}

DIALOG_GEN_PROMPT_TEMPLATE = """Here is a video description of an user working on the task - {goal_description}:
{step_descriptions}

Note: Visual information describing what the user is seeing is embedded in the descriptions as "(image shows: ...)". The user should ask visual-based questions referring to what they observe.

Your goal is to simulate a conversation between the user and an assistant, where the user's actions are performed following the assistant's instructions. The user SHOULD first mention the overall goal of the task. The assistant informs the user about the next step at proper time. Importantly, the assistant is proactive and always provides the next step even before the user asks for it. Before the task starts, the assistant may also give a brief introduction about the task. {additional_requirement}

Requirements for the assistant:
- Time is crucial! Try to generate the dialog that strictly aligns with the video timeline.
- Try to cover all the essential steps in the task. If the user asks a question at the time the assistant should give the next step, the assistant turn should include both the response to the question and instruction about the next step.
- Be helpful and friendly. If the user asks something that has been explained before, the assistant should still provide the information with patience.
- Try to be encouraging when the user makes progress, but do not overdo it.
- Be concise! The dialog is verbal, so avoid long sentences.
- Do not say "can you do it for me" to the user.
- When responding to visual questions, reference the visual information from "(image shows: ...)" descriptions to provide context-aware answers.
- Avoid redundant instructions that can be easily inferred from context using common sense. Treat the user as a fully functional adult. Focus on essential instructions only.
  * Example: Instead of "Pick up the tea leaves container from the shelf and open it. Pick up the measuring spoon from inside the tea leaves container. Scoop some tea leaves from the container with the measuring spoon and pour them into the pot on the stove top.", say "Pick up the tea leaves container from the shelf. Then use the measuring spoon inside the container to scoop some tea leaves and pour them into the pot on the stove top."
  * Example: Instead of "Place your left hand on the stove knob and turn off the stove.", say "Turn off the stove by turning the knob."


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

### New prompts for frame-aware dialog generation

DIALOG_GEN_PROMPT_TEMPLATE_WITH_VIDEO = """Here is a video description of an user working on the task - {goal_description}:
{step_descriptions}

A video clip showing this part of the task from the user perspective is also provided for your reference. The user should ask visual-based questions referring to what they observe.

Your goal is to simulate a conversation between the user and an assistant, where the user's actions are performed following the assistant's instructions. The user will first mention the overall goal of the task. The assistant informs the user about the next step at proper time. Importantly, the assistant is proactive and always provides the next step even before the user asks for it. Before the task starts, the assistant may also give a brief introduction about the task. {additional_requirement}

Requirements for the assistant:
- Time is crucial! Try to generate the dialog that strictly aligns with the video timeline.
- Try to cover all the essential steps in the task. If the user asks a question at the time the assistant should give the next step, the assistant turn should include both the response to the question and instruction about the next step.
- Be helpful and friendly. If the user asks something that has been explained before, the assistant should still provide the information with patience.
- Try to be encouraging when the user makes progress, but do not overdo it.
- Be concise! The dialog is verbal, so avoid long sentences.
- Do not say "can you do it for me" to the user.
- When responding to visual questions, use the provided video clip to give accurate, context-aware answers about visual states and progress.
- Avoid redundant instructions that can be easily inferred from context using common sense. Treat the user as a fully functional adult. Focus on essential instructions only.
  * Example: Instead of "Pick up the tea leaves container from the shelf and open it. Pick up the measuring spoon from inside the tea leaves container. Scoop some tea leaves from the container with the measuring spoon and pour them into the pot on the stove top.", say "Pick up the tea leaves container from the shelf. Then use the measuring spoon inside the container to scoop some tea leaves and pour them into the pot on the stove top."
  * Example: Instead of "Place your left hand on the stove knob and turn off the stove.", say "Turn off the stove by turning the knob."


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

DIALOG_GEN_PROMPT_TEMPLATE_WITH_FRAMES = """Here is a video description of an user working on the task - {goal_description}:
{step_descriptions}

Key frames from the video in the user's perspective are provided alongside the descriptions to give you visual context. The user should ask visual-based questions referring to what they observe.

Your goal is to simulate a conversation between the user and an assistant, where the user's actions are performed following the assistant's instructions. The user will first mention the overall goal of the task. The assistant informs the user about the next step at proper time. Importantly, the assistant is proactive and always provides the next step even before the user asks for it. Before the task starts, the assistant may also give a brief introduction about the task. {additional_requirement}

Requirements for the assistant:
- Time is crucial! Try to generate the dialog that strictly aligns with the video timeline.
- Try to cover all the essential steps in the task. If the user asks a question at the time the assistant should give the next step, the assistant turn should include both the response to the question and instruction about the next step.
- Be helpful and friendly. If the user asks something that has been explained before, the assistant should still provide the information with patience.
- Try to be encouraging when the user makes progress, but do not overdo it.
- Be concise! The dialog is verbal, so avoid long sentences.
- Do not say "can you do it for me" to the user.
- When responding to visual questions, reference the provided key frames to give accurate, context-aware answers about visual states, colors, textures, and cooking progress.
- Avoid redundant instructions that can be easily inferred from context using common sense. Treat the user as a fully functional adult. Focus on essential instructions only.
  * Example: Instead of "Pick up the tea leaves container from the shelf and open it. Pick up the measuring spoon from inside the tea leaves container. Scoop some tea leaves from the container with the measuring spoon and pour them into the pot on the stove top.", say "Pick up the tea leaves container from the shelf. Then use the measuring spoon inside the container to scoop some tea leaves and pour them into the pot on the stove top."
  * Example: Instead of "Place your left hand on the stove knob and turn off the stove.", say "Turn off the stove by turning the knob."


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

FRAME_DESCRIPTION_PROMPT = """You are viewing an egocentric (first-person) video frame of someone cooking in a kitchen. Describe the most important visual information that would help a kitchen assistant provide better guidance. In 2-3 concise sentences (30-50 words), focus on: (1) Key ingredients, tools, or cooking vessels visible and their state (e.g., 'pan is heating', 'knife is on the cutting board', 'onions are translucent') and (2) Any visual cues about cooking progress or technique (e.g., 'oil shimmering', 'vegetables browning', 'steam rising'). Focus on providing visual context, don't infer what they imply."""

ADDITIONAL_REQUIREMENTS = {
    # "holoassist": "Note that the video description contains both the user's actions and the user-assistant dialog. Anchor the dialog to the **key steps** of the task, not every single action of the user. Errors made by the user and the timing of original dialog can be a strong hint for when to simulate the dialog. You may rephrase the dialog to make it more coherent and human-like.", 
    "holoassist": "Note that the video description contains both the user's actions and the user-assistant dialog. Anchor the simulated dialog to the existing dialog, and try to rephrase the utterances to make them more coherent and human-like. You may add a few more turns around the **essential steps** of the task, which are the underlying intentions of the action instead of the actions themselves. Add a few turns to make the dialog more fluent and helpful, but avoid being overwhelming.",
    "egoexolearn": "The simulated dialog should be centered around the **key steps** of the task, not every single action of the user. Try to make the dialog more coherent and helpful as what a human assistant will say.",
    "egoexo4d": "Note that in the video description, letters (such as 'C', 'O', 'X') are used to identify different people in the annotations. The user is the person performing the task. The simulated dialog should be centered around the **key steps** of the task, not every single action of the user. Try to make the dialog more coherent and helpful as what a human assistant will say. Do NOT give overly granular instructions such as specifying which hand the user should use for an action or transferring certain tool to a hand - the user can decide this with common sense.",
    "epfl": "The simulated dialog should be centered around the **key steps** of the task, not every single action of the user. Try to make the dialog more coherent and helpful as what a human assistant will say.",
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

PROGRESS_SUMMARY_PROMPT_TEMPLATE = """Here is a conversation between a user and an assistant working on a cooking task:
{dialog_history}

Your task is to generate a concise progress summary that the assistant can use to maintain context. Focus on ESSENTIAL information that affects future steps.

INCLUDE:
- Current state of ingredients (e.g., "onions translucent", "chicken browned")
- What's in pans/bowls/oven and their current status
- Equipment currently in use
- Active timers and their purposes
- Important techniques used that might be referenced later
- Things that need attention soon
- User context: skill level demonstrated, dietary substitutions made
- Critical warnings or reminders (must-do timing, safety concerns)
- Visual states verifiable in recent frames (e.g., "vegetables showing light char on edges")

EXCLUDE:
- Granular atomic actions unless they're critical
- Overly detailed play-by-play of past actions
- Redundant information
- General chitchat unrelated to cooking
- Exact verbatim exchanges (paraphrase instead)
- Steps that are truly "done and dusted" with no future relevance

FORMATTING:
- Most recent steps get more detail
- Earlier steps get compressed unless they have future relevance
- Be concise but complete
- Use a structured format similar to:

DISH: [dish name]
PROGRESS: Step X/Y - [step in recipe]
CURRENT STATE: [what's happening now, what's in active use]
COMPLETED: [high-level summary of what's been done, emphasizing items with future impact]
NEXT: [what needs to happen next]
TIMERS: [any active timers or "None active"]
NOTES: [user preferences, skill observations, warnings]
VISUAL: [observable states from recent actions]

Give your response in the following format:
SUMMARY: <structured progress summary following the format above>
"""

PROGRESS_SUMMARY_WITH_KEYSTEPS_PROMPT_TEMPLATE = """Here is a conversation between a user and an assistant working on a cooking task:
{dialog_history}

Recipe:
{recipe_knowledge}

Past Key Steps:
{past_keysteps}

Current Key Step: {current_keystep}

Next Key Step: {next_keystep}

Your task is to generate a concise progress summary that the assistant can use to maintain context. Focus on ESSENTIAL information that affects future steps.

INCLUDE:
- Current recipe step number and what stage we're at (e.g., "Step 4/7: Sautéing vegetables")
- Current state of ingredients (e.g., "onions translucent", "chicken browned")
- What's in pans/bowls/oven and their current status
- Equipment currently in use
- Active timers and their purposes
- Important techniques used that might be referenced later
- Things that need attention soon
- User context: skill level demonstrated, dietary substitutions made
- Critical warnings or reminders (must-do timing, safety concerns)
- Visual states verifiable in recent frames (e.g., "vegetables showing light char on edges")

EXCLUDE:
- Granular atomic actions unless they're critical
- Overly detailed play-by-play of past actions
- Redundant information
- General chitchat unrelated to cooking
- Exact verbatim exchanges (paraphrase instead)
- Steps that are truly "done and dusted" with no future relevance

FORMATTING:
- Most recent steps get more detail
- Earlier steps get compressed unless they have future relevance
- Be concise but complete
- Use a structured format similar to:

DISH: [dish name]
PROGRESS: Step X/Y - [step in recipe and NOT the key steps]
CURRENT STATE: [what's happening now (include current key step), what's in active use]
COMPLETED: [high-level summary of what's been done using past key steps, emphasizing items with future impact]
NEXT: [what needs to happen next based on next key step]
TIMERS: [any active timers or "None active"]
NOTES: [user preferences, skill observations, warnings]
VISUAL: [observable states from recent actions]

Give your response in the following format:
SUMMARY: <structured progress summary following the format above>
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

    print(f"         🧠 Knowledge inference strategy for '{dataset_name}'...")
    
    # generate {num_repeats} pieces of knowledge based on the video descriptions
    if dataset_name == "egoexolearn":
        # use LLM to select from GT tasks & recipes for egoexolearn
        print(f"         Using task matching from predefined EgoExoLearn tasks")
        tasks, task_descs = EGOEXOLEARN_TASKS, get_task_descriptions()
        return match_task(step_descriptions, llm, tasks, task_descs, num_repeats)
    elif dataset_name == "wtag":
        # can simply get the GT recipe by key word matching for WTaG
        print(f"         Using keyword matching for WTaG recipes")
        return get_task_and_recipe(step_descriptions)
    elif dataset_name == "ego4d":
        print(f"         Using Ego4D recipe generation template")
        gen_prompt = EGO4D_RECIPE_GEN_PROMPT_TEMPLATE.format(
            goal_description=goal_description, step_descriptions=step_descriptions
        )
    else:
        print(f"         Using general knowledge generation for {dataset_name}")
        goal = "a task" if not goal_description else f"the task - {goal_description}"
        gen_prompt = KNOWLEDGE_GEN_PROMPT_TEMPLATE.format(
            goal_description=goal,
            step_descriptions=step_descriptions,
            knowledge_type=knowledge_type,
        )
    
    print(f"         Generating {num_repeats} knowledge candidates with LLM...")
    inputs = [("system", DEFAULT_SYS_PROMPT), ("user", gen_prompt)]
    outputs = llm.generate(inputs, n=num_repeats)

    knowledges = ""
    for i, t in enumerate(outputs):
        knowledges += f"{knowledge_type.capitalize()} {i+1}:\n {t}\n\n"

    # refine the knowledges into a single correct and complete manual
    print(f"         Refining {num_repeats} candidates into final knowledge...")
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
    print(f"         Knowledge inference completed: '{inferred_goal}'")
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
    for clip_idx, (st, et, desc) in enumerate(clips):
        print(f"      📹 Clip {clip_idx+1}/{len(clips)}: {st:.1f}s-{et:.1f}s ({et-st:.1f}s duration)")
        print(f"         Description preview: {desc[:150]}...")
        
        batch_inputs = []
        for i, user_req in enumerate(user_reqs):
            dialog_history = conversation_dict_to_text(
                batch_conv[i], add_labels=True, max_turns_to_keep=20
            )
            if dialog_history:
                dialog_history = f"You have already generated the following dialog:\n{dialog_history}"
                print(f"         Using conversation history for user type {user_types[i]} ({len(batch_conv[i])} previous turns)")
            else:
                print(f"         Starting fresh conversation for user type {user_types[i]}")
                
            prompt = DIALOG_GEN_PROMPT_TEMPLATE.format(
                goal_description=goal_description,
                step_descriptions=desc,
                user_requirement=user_req,
                dialog_history=dialog_history,
                start_time=st,
                end_time=et,
                additional_requirement=additional_requirement,
            )
            batch_inputs.append([("system", DIALOG_GEN_SYS_PROMPT), ("user", prompt)])

        # parallel generate for all user profiles
        print(f"         🤖 Generating dialogs for {len(user_types)} user types...")
        outputs = llm.batch_generate(batch_inputs)

        # add the generated dialog to the conversation history
        clip_convs = []
        for output in outputs:
            conv_dict = parse_text_to_conversation_dict(output[0])
            conv_dict = [c for c in conv_dict if c["time"] <= et]
            clip_convs.append(conv_dict)

        print(f"         🔧 Refining and labeling generated dialogs...")
        clip_convs_refined = refine_and_label_dialog(clip_convs, llm)
        
        for idx, conv in enumerate(clip_convs_refined):
            batch_conv[idx].extend(conv)
            print(f"         Added {len(conv)} turns to conversation {idx+1} ({user_types[idx]})")

    print(f"   📊 Final conversation statistics:")
    for i, conv in enumerate(batch_conv):
        print(f"      Conversation {i+1} ({user_types[i]}): {len(conv)} total turns")

    # refine the dialogs and add assistant intention labels
    # batch_conv = refine_and_label_dialog(batch_conv, llm)
    # for idx, conv in enumerate(batch_conv):
    #     print(f"Conversation {idx}, after refinement")
    #     print(conversation_dict_to_text(conv, add_labels=True))

    conv_with_user_type = [
        {"conversation": c, "user_type": p} for c, p in zip(batch_conv, user_types)
    ]
    return conv_with_user_type


def find_arrow_file_for_video(frames_dir: str, video_uid: str, take_name: Optional[str] = None) -> Optional[str]:
    """
    Find the Arrow file for a given video.
    
    For most datasets: looks for {video_uid}.arrow
    For EgoExo4D: looks for {take_name}_downscaled_*aria*.arrow
    
    Args:
        frames_dir: Directory containing frame Arrow files
        video_uid: The video UID
        take_name: Optional take name (for EgoExo4D dataset)
        
    Returns:
        Full path to the Arrow file if found, None otherwise
    """
    if not os.path.exists(frames_dir):
        return None
    
    # First try direct match with video_uid
    direct_file = os.path.join(frames_dir, f"{video_uid}.arrow")
    if os.path.exists(direct_file):
        return direct_file
    
    # If take_name is provided, try EgoExo4D pattern: {take_name}_downscaled_*aria*.arrow
    if take_name:
        prefix = f"{take_name}_downscaled_"
        for filename in os.listdir(frames_dir):
            if filename.startswith(prefix) and "aria" in filename and filename.endswith(".arrow"):
                return os.path.join(frames_dir, filename)
    
    return None


@retry_on_failure()
def generate_conversation_with_frames(
    goal_description: str,
    clips: list[tuple[float, float, str]],
    llm: LLMGenerator,
    user_types: list[str],
    additional_requirement: str = "",
    frames_dir: Optional[str] = None,
    video_uid: str = "",
    take_name: Optional[str] = None,
    use_frames: str = "frames",  # "video" or "frames"
    frames_fps: float = 2.0,
) -> list[str]:
    """Generate conversations with visual context (video or frames).
    
    This function is similar to generate_conversation but supports multimodal input.
    For Option 1 (use_frames="video"): Creates video clips from frames for each clip.
    For Option 2 (use_frames="frames"): Includes individual frames for annotations in each clip.
    
    Note: Frame descriptions (Option 3) are handled in parse_egoexo4d_annotations,
    so this function uses the regular text-based generation.
    """
    import re
    from mmassist.datasets.generate.frame_utils import (
        load_frames_from_arrow,
        get_frame_at_timestamp,
        image_to_base64_data_url,
    )
    
    # Load frames if needed
    frames_data = None
    if frames_dir is not None and (use_frames in ["video", "frames"]):
        arrow_file = find_arrow_file_for_video(frames_dir, video_uid, take_name)
        if arrow_file and os.path.exists(arrow_file):
            try:
                frames_data = load_frames_from_arrow(arrow_file)
                print(f"      Loaded {len(frames_data)} frames from {os.path.basename(arrow_file)}")
            except Exception as e:
                print(f"      Warning: Failed to load frames from {arrow_file}: {e}")
                # Fall back to text-only generation
                use_frames = "none"
        else:
            if take_name:
                print(f"      Warning: Frame file not found for video_uid '{video_uid}' or take_name '{take_name}'")
            else:
                print(f"      Warning: Frame file not found for video_uid '{video_uid}'")
            use_frames = "none"
    
    user_reqs = [DIALOG_GEN_USER_REQUIREMENTS[p] for p in user_types]

    batch_conv = [[] for _ in range(len(user_reqs))]
    for clip_idx, (st, et, desc) in enumerate(clips):
        print(f"      📹 Clip {clip_idx+1}/{len(clips)}: {st:.1f}s-{et:.1f}s ({et-st:.1f}s duration)")
        print(f"         Description preview: {desc[:150]}...")
        
        # Extract timestamps from description for frame selection (Option 2)
        clip_timestamps = []
        if use_frames == "frames" and frames_data is not None:
            # Parse timestamps from description lines like "[14.2s] ..."
            timestamp_pattern = r'\[(\d+\.?\d*)s\]'
            matches = re.findall(timestamp_pattern, desc)
            clip_timestamps = [float(t) for t in matches]
            print(f"         Found {len(clip_timestamps)} timestamps for frame selection")
        
        batch_inputs = []
        for i, user_req in enumerate(user_reqs):
            dialog_history = conversation_dict_to_text(
                batch_conv[i], add_labels=True, max_turns_to_keep=20
            )
            if dialog_history:
                dialog_history = f"You have already generated the following dialog:\n{dialog_history}"
                print(f"         Using conversation history for user type {user_types[i]} ({len(batch_conv[i])} previous turns)")
            else:
                print(f"         Starting fresh conversation for user type {user_types[i]}")
            
            # Choose the appropriate prompt template based on mode
            if use_frames == "video":
                prompt_template = DIALOG_GEN_PROMPT_TEMPLATE_WITH_VIDEO
            elif use_frames == "frames":
                prompt_template = DIALOG_GEN_PROMPT_TEMPLATE_WITH_FRAMES
            else:
                prompt_template = DIALOG_GEN_PROMPT_TEMPLATE
                
            prompt_text = prompt_template.format(
                goal_description=goal_description,
                step_descriptions=desc,
                user_requirement=user_req,
                dialog_history=dialog_history,
                start_time=st,
                end_time=et,
                additional_requirement=additional_requirement,
            )
            
            # For Option 2 (frames): Create multimodal content with images
            if use_frames == "frames" and frames_data is not None and clip_timestamps:
                # Create multimodal content with text + frames
                content_parts = [{"type": "text", "text": prompt_text}]
                
                # Add frames for each timestamp
                frames_added = 0
                for timestamp in clip_timestamps:
                    try:
                        frame = get_frame_at_timestamp(frames_data, timestamp, frames_fps)
                        if frame is not None:
                            # Convert frame to base64 data URL
                            frame_data_url = image_to_base64_data_url(frame, format="JPEG")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": frame_data_url}
                            })
                            frames_added += 1
                    except Exception as e:
                        print(f"         Warning: Failed to get frame at {timestamp}s: {e}")
                
                if frames_added > 0:
                    print(f"         Added {frames_added} frames to multimodal prompt")
                    batch_inputs.append([("system", DIALOG_GEN_SYS_PROMPT), ("user", content_parts)])
                else:
                    # Fall back to text-only if no frames could be added
                    print(f"         Warning: No frames added, falling back to text-only")
                    batch_inputs.append([("system", DIALOG_GEN_SYS_PROMPT), ("user", prompt_text)])
            else:
                # Text-only generation for other modes or when frames unavailable
                batch_inputs.append([("system", DIALOG_GEN_SYS_PROMPT), ("user", prompt_text)])
            
            # TODO for Option 1 (video mode):
            # 1. Extract frames for this clip: get_frames_for_clip(frames_data, st, et, frames_fps)
            # 2. Convert to video: frames_to_video_bytes(clip_frames, frames_fps)
            # 3. Upload video and get URL (OpenRouter doesn't support inline video yet)

        # parallel generate for all user profiles
        print(f"         🤖 Generating dialogs for {len(user_types)} user types...")
        outputs = llm.batch_generate(batch_inputs)

        # add the generated dialog to the conversation history
        clip_convs = []
        for output in outputs:
            conv_dict = parse_text_to_conversation_dict(output[0])
            conv_dict = [c for c in conv_dict if c["time"] <= et]
            clip_convs.append(conv_dict)

        print(f"         🔧 Refining and labeling generated dialogs...")
        clip_convs_refined = refine_and_label_dialog(clip_convs, llm)
        
        for idx, conv in enumerate(clip_convs_refined):
            batch_conv[idx].extend(conv)
            print(f"         Added {len(conv)} turns to conversation {idx+1} ({user_types[idx]})")

    print(f"   📊 Final conversation statistics:")
    for i, conv in enumerate(batch_conv):
        print(f"      Conversation {i+1} ({user_types[i]}): {len(conv)} total turns")

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


def find_current_and_next_keystep(time: float, keystep_segments: list[dict]) -> tuple[Optional[dict], Optional[dict]]:
    """
    Find the current and next keystep for a given time.
    
    Args:
        time: The timestamp to check (in seconds)
        keystep_segments: List of keystep segment dictionaries with start_time, end_time, step_name, step_description, is_essential
    
    Returns:
        Tuple of (current_keystep, next_keystep), where each is a dict or None
    """
    if not keystep_segments:
        return None, None
    
    # First, try to find a keystep that contains this time
    for i, segment in enumerate(keystep_segments):
        if segment["start_time"] <= time <= segment["end_time"]:
            # Found the containing keystep, now find the next one
            next_keystep = keystep_segments[i + 1] if i + 1 < len(keystep_segments) else None
            return segment, next_keystep
    
    # If no containing keystep, find the closest one
    # Find the last keystep that ended before this time
    current_keystep = None
    for i, segment in enumerate(keystep_segments):
        if segment["end_time"] < time:
            current_keystep = segment
        else:
            # This segment starts after the time, so it might be the next one
            next_keystep = segment
            return current_keystep, next_keystep
    
    # If we're after all keysteps, the last one is current and there's no next
    return keystep_segments[-1] if keystep_segments else None, None


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


@retry_on_failure()
def add_progress_summary_with_keysteps(
    conversation: list[dict], 
    llm: LLMGenerator,
    task_knowledge: str,
    keystep_segments: list[dict]
) -> dict:
    """
    Add progress summaries to the conversation with keystep awareness.
    
    Args:
        conversation: List of conversation turns
        llm: LLM generator
        task_knowledge: The recipe/task knowledge (inferred or provided)
        keystep_segments: List of keystep annotations with start_time, end_time, step_name
    """
    batch_inputs = []
    summ_turn_ids = []
    
    # Build keysteps context string
    total_steps = len(keystep_segments)
    keysteps_list = []
    for i, seg in enumerate(keystep_segments):
        keysteps_list.append(
            f"{i+1}. {seg['step_name']} "
            f"({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)"
        )
    keysteps_context = "\n".join(keysteps_list)
    
    for idx, turn in enumerate(conversation):
        if turn["role"] == "assistant":
            time = turn["time"]
            
            # Find current and next keystep
            current_step, next_step = find_current_and_next_keystep(time, keystep_segments)
            
            # Format current and next step info
            if current_step:
                # Find step number
                step_num = next(
                    (i+1 for i, s in enumerate(keystep_segments) if s == current_step),
                    0
                )
                current_step_str = current_step['step_name']
                
                # Build past keysteps string (all completed keysteps before current)
                past_keysteps_list = []
                for i in range(step_num - 1):
                    past_keysteps_list.append(f"{i+1}. {keystep_segments[i]['step_name']}")
                past_keysteps_str = "\n".join(past_keysteps_list) if past_keysteps_list else "None (just starting)"
            else:
                current_step_str = "Not yet started or between steps"
                past_keysteps_str = "None (just starting)"
            
            if next_step:
                next_step_str = next_step['step_name']
            else:
                next_step_str = "Task nearly complete or complete"

            # print(f"Keysteps for turn at time: {time} | Past: {past_keysteps_str} | Current: {current_step_str} | Next: {next_step_str}")

            # Build dialog history
            dh = conversation_dict_to_text(conversation[: idx + 1], add_labels=False)
            
            # Build prompt
            prompt = PROGRESS_SUMMARY_WITH_KEYSTEPS_PROMPT_TEMPLATE.format(
                recipe_knowledge=task_knowledge,
                past_keysteps=past_keysteps_str,
                dialog_history=dh,
                current_keystep=current_step_str,
                next_keystep=next_step_str
            )
            
            inputs = [("system", SUMMARY_SYS_PROMPT), ("user", prompt)]
            batch_inputs.append(inputs)
            summ_turn_ids.append(idx)

    # generate the progress summary in batch
    batch_outputs = llm.batch_generate(batch_inputs)

    # update the conversation with the progress summary
    for turn_idx, outputs in zip(summ_turn_ids, batch_outputs):
        time = conversation[turn_idx]["time"]
        elsp = f"The time elapsed since the start of the task is {time:.1f} seconds.\n\n"

        # Extract the summary from the output
        output_text = outputs[0]
        if "SUMMARY:" in output_text:
            # Get everything after SUMMARY:
            summary_start = output_text.find("SUMMARY:")
            progress = elsp + output_text[summary_start + len("SUMMARY:"):].strip()
        else:
            raise ValueError(f"Failed to parse summary from output: {output_text}")

        conversation[turn_idx]["progress"] = progress

    return conversation


@dataclass
class ParsedVideoAnns:
    dataset: str
    domain: str  # "cooking", "object manipulation", "lab"
    knowledge_type: str  # "cooking recipe", ...
    video_uid: str
    goal_description: str
    all_step_descriptions: str
    clips: list[tuple[float, float, str]]
    duration: float
    ann_ratio: float
    num_steps: int
    video_start_time: float = 0.0
    has_mistake: bool = False
    num_substeps: Optional[int] = None
    fps: Optional[float] = None
    take_name: Optional[str] = None  # For EgoExo4D: used to find Arrow files with pattern {take_name}_downscaled_*aria*.arrow
    original_ann: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratedOutputs:
    video_uid: str
    inferred_goal: str
    inferred_knowledge: str
    video_labels: Counter
    conversations: List[Dict[str, Union[str, List[dict]]]]
    parsed_video_anns: Optional[dict] = None

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
    frames_dir: Optional[str] = None,
    use_frames: str = "none",
    frames_fps: float = 2.0,
    use_keysteps: bool = False,
) -> Union[GeneratedOutputs, str]:

    video_uid = annotation.video_uid
    print(f"\n{'='*80}")
    print(f"🎬 PROCESSING VIDEO: {video_uid}")
    print(f"{'='*80}")

    dataset = annotation.dataset
    knowledge_type = annotation.knowledge_type
    goal_description = annotation.goal_description
    step_descriptions = annotation.all_step_descriptions
    ann_ratio = annotation.ann_ratio
    
    # Step 0: Input validation
    print(f"📋 STEP 0: Input Validation")
    print(f"   Dataset: {dataset}")
    print(f"   Knowledge type: {knowledge_type}")
    print(f"   Annotation ratio: {ann_ratio:.2f} (min required: {min_ann_ratio})")
    
    if ann_ratio < min_ann_ratio:
        skip_msg = f"❌ Skip video {video_uid} with low annotation ratio: {ann_ratio}"
        print(skip_msg)
        return skip_msg

    print(f"✅ Video passed validation")
    print(
        (
            f"   Goal: {goal_description}\n"
            f"   Duration: {annotation.duration:.1f}s | "
            f"   Num steps: {annotation.num_steps} | "
            f"   Num substeps: {annotation.num_substeps} | "
            f"   Num clips: {len(annotation.clips)} | "
            f"   Ann ratio: {annotation.ann_ratio:.2f}"
        )
    )

    # 1. infer goal and recipe
    print(f"\n📚 STEP 1: Goal & Knowledge Inference")
    print(f"   Strategy: {dataset}-specific inference")
    print(f"   Original goal: '{goal_description}'")
    print(f"   Generating {num_repeats} knowledge candidates...")
    
    inferred_goal, inferred_knowledge = infer_goal_and_knowledge(
        dataset, goal_description, step_descriptions, knowledge_type, llm, num_repeats
    )
    
    print(f"✅ Knowledge inference completed")
    print(f"   Inferred goal: '{inferred_goal}'")
    print(f"   Inferred knowledge preview: {inferred_knowledge[:200]}...")
    
    if use_inferred_goal:
        print(f"   🔄 Using inferred goal instead of original")
        goal_description = inferred_goal
    else:
        print(f"   📌 Keeping original goal description")

    # 2. label video and filter out inappropriate videos
    print(f"\n🏷️  STEP 2: Video Labeling & Filtering")
    
    if filter_by_llm:
        print(f"   Enabled: Running LLM-based video filtering with {num_repeats} evaluations")
        video_labels = label_video(
            goal_description,
            step_descriptions,
            inferred_knowledge,
            knowledge_type,
            domain=annotation.domain,
            llm=llm,
            num_repeats=num_repeats,
        )
        print(f"   Label distribution: {dict(video_labels)}")
        label, cnt = video_labels.most_common(1)[0]
        print(f"   Most common label: {label} (count: {cnt}/{num_repeats})")
        
        if label != 1 or cnt < num_repeats // 2:
            skip_msg = f"❌ Skip video {video_uid} - failed LLM filtering: {video_labels}"
            print(skip_msg)
            return skip_msg
        print(f"✅ Video passed LLM filtering")
        video_labels = dict(video_labels)
    else:
        print(f"   Disabled: Skipping LLM-based filtering")
        video_labels = {}

    # 3. generate the user-assistant conversations
    print(f"\n💬 STEP 3: Conversation Generation")
    print(f"   User types: {user_types}")
    print(f"   Number of clips to process: {len(annotation.clips)}")
    print(f"   Frame mode: {use_frames}")
    
    clips = annotation.clips
    add_reqs = ADDITIONAL_REQUIREMENTS.get(annotation.dataset, "")
    if dataset == "assembly101" and not annotation.has_mistake:
        add_reqs = ""
    
    if add_reqs:
        print(f"   Dataset-specific requirements: {add_reqs[:100]}...")
    else:
        print(f"   No additional dataset-specific requirements")
        
    print(f"   🔄 Starting clip-by-clip dialog generation...")
    
    # Choose generation function based on frame mode
    # Note: Option 3 (descriptions) uses regular generate_conversation since
    # descriptions are already embedded in the text
    if use_frames in ["video", "frames"]:
        conversations = generate_conversation_with_frames(
            goal_description, clips, llm, user_types, add_reqs,
            frames_dir=frames_dir,
            video_uid=video_uid,
            take_name=annotation.take_name,  # For EgoExo4D: helps find Arrow files
            use_frames=use_frames,
            frames_fps=frames_fps,
        )
    else:
        conversations = generate_conversation(
            goal_description, clips, llm, user_types, add_reqs
        )
    
    print(f"✅ Generated {len(conversations)} conversations")
    
    # adjust time
    print(f"   🕐 Adjusting conversation timing...")
    for i, conv in enumerate(conversations):
        original_turns = len(conv["conversation"])
        conv["conversation"] = adjust_time(conv["conversation"])
        print(f"      Conversation {i+1}: {original_turns} turns, timing adjusted")

    # 4. add progress summary
    print(f"\n📝 STEP 4: Progress Summary Generation")
    if use_keysteps:
        print(f"   Using keystep-aware progress summaries...")
        
        # Extract keystep segments from original_ann
        keystep_segments = []
        if annotation.original_ann and "keystep_annotations" in annotation.original_ann:
            keystep_anns = annotation.original_ann["keystep_annotations"]
            if "segments" in keystep_anns:
                keystep_segments = keystep_anns["segments"]
                print(f"   Found {len(keystep_segments)} keystep segments")
            else:
                print(f"   ⚠️  Warning: No segments in keystep_annotations, falling back to regular summaries")
                use_keysteps = False
        else:
            print(f"   ⚠️  Warning: No keystep_annotations found in annotation, falling back to regular summaries")
            use_keysteps = False
        
        if use_keysteps:
            # Use the inferred knowledge as task knowledge
            task_knowledge = inferred_knowledge
            
            for i, conv in enumerate(conversations):
                assistant_turns_before = sum(1 for turn in conv["conversation"] if turn["role"] == "assistant")
                conv["conversation"] = add_progress_summary_with_keysteps(
                    conv["conversation"], 
                    llm, 
                    task_knowledge, 
                    keystep_segments
                )
                assistant_turns_after = sum(1 for turn in conv["conversation"] if turn["role"] == "assistant" and "progress" in turn)
                print(f"      Conversation {i+1}: Added keystep-aware summaries to {assistant_turns_after}/{assistant_turns_before} assistant turns")
    
    if not use_keysteps:
        print(f"   Adding standard progress summaries to assistant turns...")
        for i, conv in enumerate(conversations):
            assistant_turns_before = sum(1 for turn in conv["conversation"] if turn["role"] == "assistant")
            conv["conversation"] = add_progress_summary(conv["conversation"], llm)
            assistant_turns_after = sum(1 for turn in conv["conversation"] if turn["role"] == "assistant" and "progress" in turn)
            print(f"      Conversation {i+1}: Added summaries to {assistant_turns_after}/{assistant_turns_before} assistant turns")

    print(f"✅ Progress summaries completed")

    # return the generated outputs
    print(f"\n🎯 STEP 5: Output Preparation")
    print(f"   Packaging results for video {video_uid}")
    print(f"   Generated conversations: {len(conversations)}")
    for i, conv in enumerate(conversations):
        turns = len(conv["conversation"])
        user_turns = sum(1 for turn in conv["conversation"] if turn["role"] == "user")
        assistant_turns = sum(1 for turn in conv["conversation"] if turn["role"] == "assistant")
        print(f"      Conversation {i+1} ({conv['user_type']}): {turns} total turns ({user_turns} user, {assistant_turns} assistant)")
    
    outputs = GeneratedOutputs(
        video_uid=video_uid,
        inferred_goal=inferred_goal,
        inferred_knowledge=inferred_knowledge,
        video_labels=video_labels,
        conversations=conversations,
        parsed_video_anns=annotation.to_dict() if keep_original_anns else None,
    )
    
    print(f"✅ Successfully generated dialog for video {video_uid}")
    print(f"{'='*80}\n")
    return outputs
