import os
from mmassist.eval.eval import EvalArguments, run_eval
from mmassist.configs.arguments import DATA_ROOT_DIR

model_root = f"{DATA_ROOT_DIR}/models"

for exp_name, run_type, notalk_rates in [
    ########################### Narration ###########################
    # (
    #     "20240816-L4096-I1-ep4-NOSEP-nr1.0-1s-lora-bs256",
    #     "narration",
    #     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # ),
    # (
    #     "20240816-L4096-I1-ep4-NOSEP-nr0.2-1s-lora-bs256",
    #     "narration",
    #     [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.1-1s-lora-bs256",
    #     "narration",
    #     [0.2, 0.3, 0.4, 0.5],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.01-1s-lora-bs256",
    #     "narration",
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.1-w2t_head_w0.2-1s-lora-bs256",
    #     "narration",
    #     [0.2, 0.3, 0.4],
    # ),
    # (
    #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
    #     "narration",
    #     [0.2, 0.3, 0.4],
    # ),
    # (
    #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
    #     "narration",
    #     [0.2, 0.3, 0.4],
    # ),
    ########################### Dialog ###########################
    # ("20240816-L4096-I1-ep4-NOSEP-nr1.0-1s-lora-bs256", "dialog", [0.5, 0.6, 0.7, 0.8]),
    # (
    #     "20240816-L4096-I1-ep4-NOSEP-nr0.2-1s-lora-bs256",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.1-1s-lora-bs256",
    #     "dialog",
    #     [0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.01-1s-lora-bs256",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    # ),
    # (
    #     "20240820-L4096-I1-ep4-NOSEP-nr0.1-w2t_head_w0.2-1s-lora-bs256",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.4],
    # ),
    # (
    #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.35, 0.4],
    # ),
    # (
    #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.35, 0.4],
    # ),
    # (
    #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
    #     "dialog",
    #     [0.2, 0.3, 0.4, 0.5],
    # ),
    # (
    #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    # ),
    # (
    #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
    #     "dialog",
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    # ),
]:
    inference_setups = []
    # for notalk in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # for notalk in [0.3, 0.5, 0.7]:

    L = 4096
    maxlen = "4k"
    if "I1-" in exp_name:
        I = 1
    elif "I5-" in exp_name:
        I = 5
    elif "I10-" in exp_name:
        I = 10
    else:
        raise ValueError(f"Invalid exp_name: {exp_name}")
    bs = 256 if I < 10 else 512
    for notalk in notalk_rates:

        # runner = "stream"
        runner = "stream"
        if run_type == "narration":
            inference_setups.append(
                f"ego4d/narration_val_L{L}_I{I}|{runner}|{maxlen}|{notalk}|summarize_and_drop"
            )
        else:
            for data_name in [
                "ego4d",
                "holoassist",
                "epickitchens",
                "egoexolearn",
                "wtag",
                "assembly101",
            ]:
                inference_setups.append(
                    f"{data_name}/dialog_val_L0_I{I}|{runner}|{maxlen}|{notalk}|summarize_and_drop"
                )
    inference_setups = ",".join(inference_setups)

    model_path = os.path.join(model_root, exp_name)
    args = EvalArguments(
        model_path=model_path, inference_setups=inference_setups, force_rerun=False
    )

    print(f"\n\n{exp_name}\n\n")
    run_eval(args, mode="compute_metrics", verbose=False, rerun_match=True)
