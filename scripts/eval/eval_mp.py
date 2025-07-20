import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch.multiprocessing as mp
from mmassist.eval.eval import EvalArguments, run_eval
from mmassist.configs.arguments import DATA_ROOT_DIR


if __name__ == "__main__":
    mp.set_start_method("spawn")

    model_root = f"{DATA_ROOT_DIR}/models"
    num_gpus = 8

    for exp_name, inference_setups in [
        # test set
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/narration_test_L4096_I1|stream|4k|0.3|summarize_and_drop",
        #         "ego4d/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I1|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_test_L0_I1|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "assembly101/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_test_L0_I1|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # # I=5
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/narration_test_L4096_I5|stream|4k|0.3|summarize_and_drop",
        #         "ego4d/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I5|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I5|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # # I=5 (klg)
        (
            "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
            [
                "ego4d/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
                "epickitchens/dialog-klg_test_L0_I5|stream|4k|0.2|summarize_and_drop",
                "holoassist/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
                "egoexolearn/dialog-klg_test_L0_I5|stream|4k|0.4|summarize_and_drop",
                "assembly101/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
                "wtag/dialog-klg_test_L0_I5|stream|4k|0.5|summarize_and_drop",
            ],
        ),
        # # I=10
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/narration_test_L4096_I10|stream|4k|0.3|summarize_and_drop",
        #         "ego4d/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I10|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # I=10 (klg)
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog-klg_test_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_test_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "holoassist/dialog-klg_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "egoexolearn/dialog-klg_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog-klg_test_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        ########################### Dialog - w/ knowledge ###########################
        # "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        # [
        #     "ego4d/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     "epickitchens/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #     "holoassist/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     "egoexolearn/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     "assembly101/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     "wtag/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        # ],
        # [
        #     "ego4d/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #     "epickitchens/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     "holoassist/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #     "egoexolearn/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #     "assembly101/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #     "wtag/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        # ],
        # (
        #     "20240816-L4096-I1-ep4-NOSEP-nr0.2-1s-lora-bs256",
        #     [
        #         "ego4d/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #         "epickitchens/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #         "holoassist/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #         "egoexolearn/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #         "assembly101/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #         "wtag/dialog_val_L0_I1|stream|4k|0.6|summarize_and_drop",
        #     ],
        # ),
        ########################### Narration ###########################
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/narration_val_L4096_I5|stream|4k|0.2|summarize_and_drop",
        #         "ego4d/narration_val_L4096_I5|stream|4k|0.3|summarize_and_drop",
        #         "ego4d/narration_val_L4096_I5|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-pose-1s-lora-bs512",
        #     [
        #         "ego4d/narration_val_L4096_I10|stream|4k|0.2|summarize_and_drop",
        #         "ego4d/narration_val_L4096_I10|stream|4k|0.3|summarize_and_drop",
        #         "ego4d/narration_val_L4096_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        ########################### Dialog (context handling) ###########################
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog_val_L0_I10|stream|4k|0.4|drop_middle",
        #         "epickitchens/dialog_val_L0_I10|stream|4k|0.2|drop_middle",
        #         "holoassist/dialog_val_L0_I10|stream|4k|0.4|drop_middle",
        #         "egoexolearn/dialog_val_L0_I10|stream|4k|0.4|drop_middle",
        #         "assembly101/dialog_val_L0_I10|stream|4k|0.3|drop_middle",
        #         "wtag/dialog_val_L0_I10|stream|4k|0.4|drop_middle",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "wtag/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop",
        #         "assembly101/dialog_val_L0_I10|stream|32k|0.3|summarize_and_drop",
        #         # "ego4d/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop",
        #         # "epickitchens/dialog_val_L0_I10|stream|32k|0.2|summarize_and_drop",
        #         # "holoassist/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop",
        #         # "egoexolearn/dialog_val_L0_I10|stream|32k|0.4|summarize_and_drop",
        #     ],
        # ),
        ## offline
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/narration_val_L4096_I1|offline|4k|0.3|none",
        #         # "ego4d/dialog_val_L0_I1|offline|4k|0.3|none",
        #         # "epickitchens/dialog_val_L0_I1|offline|4k|0.2|none",
        #         # "holoassist/dialog_val_L0_I1|offline|4k|0.3|none",
        #         # "egoexolearn/dialog_val_L0_I1|offline|4k|0.3|none",
        #         # "assembly101/dialog_val_L0_I1|offline|4k|0.3|none",
        #         # "wtag/dialog_val_L0_I1|offline|4k|0.3|none",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/dialog-klg_val_L0_I1|offline|4k|0.3|none",
        #         "epickitchens/dialog-klg_val_L0_I1|offline|4k|0.2|none",
        #         "holoassist/dialog-klg_val_L0_I1|offline|4k|0.3|none",
        #         "egoexolearn/dialog-klg_val_L0_I1|offline|4k|0.3|none",
        #         "assembly101/dialog-klg_val_L0_I1|offline|4k|0.3|none",
        #         "wtag/dialog-klg_val_L0_I1|offline|4k|0.4|none",
        #     ],
        # ),
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         # "ego4d/narration_val_L4096_I5|offline|4k|0.3|none",
        #         # "ego4d/dialog_val_L0_I5|offline|4k|0.3|none",
        #         # "epickitchens/dialog_val_L0_I5|offline|4k|0.3|none",
        #         # "holoassist/dialog_val_L0_I5|offline|4k|0.3|none",
        #         # "egoexolearn/dialog_val_L0_I5|offline|4k|0.4|none",
        #         # "assembly101/dialog_val_L0_I5|offline|4k|0.3|none",
        #         # "wtag/dialog_val_L0_I5|offline|4k|0.4|none",
        #     ],
        # ),
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/dialog-klg_val_L0_I5|offline|4k|0.3|none",
        #         "epickitchens/dialog-klg_val_L0_I5|offline|4k|0.2|none",
        #         "holoassist/dialog-klg_val_L0_I5|offline|4k|0.3|none",
        #         "egoexolearn/dialog-klg_val_L0_I5|offline|4k|0.4|none",
        #         "assembly101/dialog-klg_val_L0_I5|offline|4k|0.3|none",
        #         "wtag/dialog-klg_val_L0_I5|offline|4k|0.5|none",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         ### narration
        #         # "ego4d/narration_val_L4096_I10|offline|4k|0.3|none",
        #         ### dialog
        #         # "ego4d/dialog_val_L0_I10|offline|4k|0.4|none",
        #         # "epickitchens/dialog_val_L0_I10|offline|4k|0.2|none",
        #         # "holoassist/dialog_val_L0_I10|offline|4k|0.4|none",
        #         # "egoexolearn/dialog_val_L0_I10|offline|4k|0.4|none",
        #         # "assembly101/dialog_val_L0_I10|offline|4k|0.3|none",
        #         # "wtag/dialog_val_L0_I10|offline|4k|0.4|none",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog-klg_val_L0_I10|offline|4k|0.3|none",
        #         "epickitchens/dialog-klg_val_L0_I10|offline|4k|0.3|none",
        #         "holoassist/dialog-klg_val_L0_I10|offline|4k|0.4|none",
        #         "egoexolearn/dialog-klg_val_L0_I10|offline|4k|0.4|none",
        #         "assembly101/dialog-klg_val_L0_I10|offline|4k|0.3|none",
        #         "wtag/dialog-klg_val_L0_I10|offline|4k|0.4|none",
        #     ],
        # ),
    ]:
        inference_setups = ",".join(inference_setups)
        model_path = os.path.join(model_root, exp_name)
        args = EvalArguments(
            model_path=model_path, inference_setups=inference_setups, force_rerun=False
        )

        mode = "local_mp_inference"
        input_args = [[args, mode, i == 0, i, num_gpus] for i in range(num_gpus)]
        with mp.Pool(num_gpus) as p:
            p.starmap(run_eval, input_args)

        print(f"\n\n{exp_name}\n\n")
        run_eval(args, mode="compute_metrics", verbose=True)
