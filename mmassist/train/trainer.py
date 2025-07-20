import os
import torch

from transformers import Trainer


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


class CustomTrainer(Trainer):
    # def prediction_step(
    #     self,
    #     model: torch.nn.Module,
    #     inputs: dict,
    #     prediction_loss_only: bool,
    #     ignore_keys: list[str] = None,
    # ):
    #     with torch.no_grad():
    #         inputs = self._prepare_inputs(inputs)
    #         with self.compute_loss_context_manager():
    #             loss = self.compute_loss(model, inputs, return_outputs=False)
    #         if prediction_loss_only:
    #             return (loss, None, None)
    #         preds = model.stream_evaluate(**inputs).reshape(1, -1)
    #         return (loss, preds, inputs.get("labels", preds))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dict = {}
        if self.model.config.use_binary_decision_head:
            if self.args.should_log:
                print("Logging w2t_loss")
                self.log_dict["w2t_loss"] = 1.0

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        try:
            if isinstance(outputs, dict):
                self.log_dict.update(outputs["log_dict"])
            elif isinstance(outputs[-1], dict):
                self.log_dict.update(outputs[-1])
        except:
            pass
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        if self.is_local_process_zero() and self.model.training:
            logs.update(self.log_dict)
        return super().log(logs)

    def _save_checkpoint(self, model, trial, metrics=None):
        # if self.args.llm_train_mode != "lora":
        #     from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        #     run_dir = self._get_output_dir(trial=trial)
        #     output_dir = os.path.join(run_dir, checkpoint_folder)

        #     # Only save the fine-tuned modules
        #     keys_to_match = self.args.finetune_modules.split(",")
        #     weight_to_save = get_mm_adapter_state_maybe_zero_3(
        #         self.model.named_parameters(), keys_to_match
        #     )
        #     self.model.config.save_pretrained(output_dir)
        #     save_name = os.path.join(output_dir, f"mm_projector.bin")
        #     torch.save(weight_to_save, save_name)

        #     # NOTE: we don't save the optimizer, scheduler and the RNG states here as
        #     # the training is fast so we don't expect to resume from any checkpoint

        # if self.args.llm_train_mode != "frozen":
        #     super()._save_checkpoint(model, trial, metrics)
        return

    def _save(self, output_dir: str | None = None, state_dict=None):
        if not self.args.should_save:
            return
        if self.args.llm_train_mode != "lora":
            # Only save the fine-tuned modules
            keys_to_match = self.args.finetune_modules.split(",")
            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )
            self.model.config.save_pretrained(output_dir)
            save_name = os.path.join(output_dir, f"mm_projector.bin")
            # FIXME: the mm_projector.bin contains all the fine-tuned parameters, not
            # just the mm_projector (e.g. the binary_decision_head if used)
            torch.save(weight_to_save, save_name)

        if self.args.llm_train_mode != "frozen":
            super()._save(output_dir, state_dict)
