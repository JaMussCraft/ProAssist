ego4d_narration_system_prompts = [
    "Describe the actions of the camera wearer (denoted as 'C').",
    'Describe the ongoing actions of the camera wearer "C".',
    "Explain what the camera wearer C is doing.",
    "Make real-time narrations of the camera wearer C's actions.",
    "Narrate what the person C wearing the camera is currently doing.",
    "Report on the ongoing activities of the camera wearer C.",
    "List any actions the camera wearer 'C' is having.",
    "Tell me what actions the person with the camera (denoted as C) is performing.",
]

proactive_assistant_system_prompts = [
    "You are a proactive assistant. Pay close attention to the user's actions and provide relevant information proactively.",
    "You are a proactive assistant. Predict the user's needs and provide assistance before being requested.",
    "You are a responsive and proactive assistant. Address questions quickly and offer assistance before it's needed.",
    "You are a helpful and proactive assistant. Always be ready to assist and provide useful information ahead of time.",
    "You are a helpful assistant. Be responsive to the user's questions, and provide useful information proactively when needed.",
    "You are a supportive assistant. Always be ready to help and offer useful advice without being asked.",
]


image_short_description_prompts = [
    "Describe each image.",
    "Describe each image in a few words.",
    "Provide a brief description of each frame.",
    "Give a short description of each image.",
    "Describe the content of each image briefly.",
    "Provide a concise description of each image.",
]

object_detection_prompts = [
    "List the objects in each image.",
    "Identify the objects in each image.",
    "List the objects present in each image.",
    "Name the objects in each image.",
    "Identify the objects visible in each image.",
    "List the objects that appear in each image.",
]

action_recognition_prompts = [
    "Describe the action happening in the video clips.",
    "Provide a description of the action taking place in each video.",
    "Describe the ongoing action in each video clip.",
    "Narrate the actions in each video clip.",
    "Explain the activities in the video clips.",
]

summarize_sys_prompt = "Watch the user's actons and track the task progress."
summarize_query_content = "Please summarize the progress."
summarize_query = {"role": "system", "content": summarize_query_content}
