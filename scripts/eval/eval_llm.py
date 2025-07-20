import os

os.environ["OMP_NUM_THREADS"] = "1"

from mmassist.eval.eval import EvalArguments, run_eval
from mmassist.datasets.generate.llm_utils import LLMGenerator
from mmassist.configs.arguments import DATA_ROOT_DIR


if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() >= 8, "Running LLM evaluation requires 8 GPUs"

    model_root = f"{DATA_ROOT_DIR}/models"


    llm_id: str = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    llm = LLMGenerator.build(model_id=llm_id, number_gpus=8)

    for exp_name, inference_setups in [
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/dialog_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "assembly101/dialog_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/dialog_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "holoassist/dialog_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_val_L0_I5|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_val_L0_I5|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "epickitchens/dialog_val_L0_I10|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "egoexolearn/dialog_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_val_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        #### with knowledge
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_val_L0_I1|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "assembly101/dialog-klg_val_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_val_L0_I1|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/dialog-klg_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_val_L0_I5|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog-klg_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog-klg_val_L0_I5|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog-klg_val_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_val_L0_I5|stream|4k|0.5|summarize_and_drop",
        #     ],
        # ),
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog-klg_val_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_val_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "holoassist/dialog-klg_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "egoexolearn/dialog-klg_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog-klg_val_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_val_L0_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        #### test
        # (
        #     "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
        #     [
        #         "ego4d/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I1|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I1|stream|4k|0.3|summarize_and_drop",
        #     ],
        # ),
        (
            "20240821-L4096-I1-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs256",
            [
                "wtag/dialog-klg_test_L0_I1|stream|4k|0.4|summarize_and_drop",
                "assembly101/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
                "egoexolearn/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
                "holoassist/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
                "epickitchens/dialog-klg_test_L0_I1|stream|4k|0.2|summarize_and_drop",
                "ego4d/dialog-klg_test_L0_I1|stream|4k|0.3|summarize_and_drop",
            ],
        ),
        # # I=5
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I5|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I5|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # I=5 (klg)
        # (
        #     "20240822-L4096-I5-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs384-debug",
        #     [
        #         "ego4d/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "epickitchens/dialog-klg_test_L0_I5|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "egoexolearn/dialog-klg_test_L0_I5|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog-klg_test_L0_I5|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog-klg_test_L0_I5|stream|4k|0.5|summarize_and_drop",
        #     ],
        # ),
        # # I=10
        # (
        #     "20240821-L4096-I10-ep4-NOSEP-nr0.1-klgmix-1s-lora-bs512",
        #     [
        #         "ego4d/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "epickitchens/dialog_test_L0_I10|stream|4k|0.2|summarize_and_drop",
        #         "holoassist/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "egoexolearn/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #         "assembly101/dialog_test_L0_I10|stream|4k|0.3|summarize_and_drop",
        #         "wtag/dialog_test_L0_I10|stream|4k|0.4|summarize_and_drop",
        #     ],
        # ),
        # # I=10 (klg)
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
    ]:
        inference_setups = ",".join(inference_setups)
        model_path = os.path.join(model_root, exp_name)
        args = EvalArguments(
            model_path=model_path, inference_setups=inference_setups, force_rerun=False
        )
        print(f"\n\n{exp_name}\n\n")

        run_eval(args, mode="llm_eval", llm=llm, verbose=True)
