import os
import logging
import torch
from dataclasses import asdict
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel

from mmassist.configs import TrainingArguments, ModelArguments
from mmassist.model.modeling_proact import ProActLlamaForCausalLM, ProActLlamaConfig
from mmassist.model.tokenization_proact import (
    build_tokenizer_and_update_config,
    AutoTokenizer,
)


def get_lora_config(train_args: TrainingArguments) -> LoraConfig:
    ft_modules = train_args.finetune_modules
    modules_to_save = ft_modules.split(",") if ft_modules else None
    return LoraConfig(
        r=train_args.lora_r,
        lora_alpha=train_args.lora_alpha,
        target_modules=train_args.lora_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    all_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}"
    )


def build_from_pretrained(
    model_args: ModelArguments, train_args: TrainingArguments
) -> tuple[ProActLlamaForCausalLM, AutoTokenizer]:
    """Build from a pretrained LLM model."""

    config = ProActLlamaConfig.from_pretrained_llama(**asdict(model_args))

    if train_args.is_debug:
        # load a smaller model for debugging
        config.hidden_size = 1024
        config.num_hidden_layers = 1
        model = ProActLlamaForCausalLM(config=config).to(torch.bfloat16)
    else:
        model: ProActLlamaForCausalLM = ProActLlamaForCausalLM.from_pretrained(
            model_args.llm_pretrained,
            config=config,
            torch_dtype="auto",
            attn_implementation=model_args.attn_implementation,
        )
    model.init_multimodal_modules()

    # update config
    model.config.architectures = [model.__class__.__name__]
    model.config.torch_dtype = str(model.dtype).split(".")[1]
    model.config.training_args = train_args.to_dict()

    # build tokenizer
    tokenizer = build_tokenizer_and_update_config(model.config)
    if train_args.llm_train_mode == "lora":
        lora_config = get_lora_config(train_args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif train_args.llm_train_mode == "full":
        model.requires_grad_(True)
        print_trainable_parameters(model)
    elif train_args.llm_train_mode == "frozen":
        model.requires_grad_(False)
        model.mm_projector.requires_grad_(True)
        if model.hasattr("binary_decision_head"):
            model.binary_decision_head.requires_grad_(True)
        print_trainable_parameters(model)

    if train_args.output_dir and train_args.local_rank in [-1, 0]:
        model.config.save_pretrained(train_args.output_dir)

    return model, tokenizer


def update_config(
    saved_config: ProActLlamaConfig,
    is_training: bool = False,
    model_args: ModelArguments | None = None,
    train_args: TrainingArguments | None = None,
) -> tuple[ProActLlamaConfig, LoraConfig | None]:
    """Update the config."""

    # update model config from model_args
    if model_args is not None:
        for key, value in model_args.to_dict().items():
            ori_value = getattr(saved_config, key)
            if ori_value != value:
                print(f"Update model config: {key} from {ori_value} to {value}")
                setattr(saved_config, key, value)

    # update train config from train_args
    lora_config = None
    if is_training and train_args is not None:
        ori_train_args = saved_config.training_args
        for key, value in train_args.to_dict().items():
            ori_value = ori_train_args.get(key)
            if ori_value != value:
                print(f"Update training args: {key} from {ori_value} to {value}")
                ori_train_args[key] = value

        if train_args.llm_train_mode == "lora":
            lora_config = get_lora_config(train_args)

    # reinstantiate the config using the updated dict
    # NOTE: this is required for the __init__ method to be called
    config = ProActLlamaConfig.from_dict(saved_config.to_dict())

    return config, lora_config


def build_from_checkpoint(
    checkpoint_dir: str,
    is_training: bool = False,
    model_args: ModelArguments | None = None,
    train_args: TrainingArguments | None = None,
    rank: int | None = None,
) -> tuple[ProActLlamaForCausalLM, AutoTokenizer]:
    """Build from a pretrained ProAct model for continue training or evaluation.

    :param checkpoint_dir: The directory created by the HF Trainer.
    :param is_training: Whether the model is for training.
    :param model_args: The model arguments to update the config.
    :param train_args: The training arguments to update the config.

    :return: The model and tokenizer.
    """

    rank = rank if rank is not None else 0
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # load the saved config for the checkpoint
    saved_config = ProActLlamaConfig.from_pretrained(checkpoint_dir)
    ckpt_llm_train_mode = saved_config.training_args["llm_train_mode"]
    llm_load_dir = (
        checkpoint_dir if ckpt_llm_train_mode == "full" else saved_config.llm_pretrained
    )

    # update the config for new model/training setups if needed
    config, new_lora_config = update_config(
        saved_config, is_training, model_args, train_args
    )

    print(f"Loading the base LLM from {llm_load_dir}")
    base_model: ProActLlamaForCausalLM = ProActLlamaForCausalLM.from_pretrained(
        llm_load_dir,
        config=config,
        torch_dtype="auto" if is_training else torch.float16,
        attn_implementation=config.attn_implementation,
        device_map=device,
        ignore_mismatched_sizes=True,
    )
    # NOTE: even with "full" training, we have not loaded the mm_projector as it is
    # not initialized yet
    base_model.init_multimodal_modules()
    # print("before load", base_model.mm_projector[0].weight.data)

    if ckpt_llm_train_mode != "lora":
        print(f"Loading the pretrained mm_projector")
        weights = torch.load(os.path.join(checkpoint_dir, "mm_projector.bin"))
        base_model.load_state_dict(weights, strict=False)
        # NOTE: mm_projector.bin should contain all the fine-tuned parameters, not
        # just the mm_projector (e.g. also the binary_decision_head if used)
        # print("after load", base_model.mm_projector[0].weight.data)
    else:
        # NOTE: w/ LoRA the mm_projector is saved with the adapter
        pass

    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    copy_adapters_to_new_output_dir = (
        is_training and train_args is not None and train_args.llm_train_mode != "full"
    )
    if os.path.exists(adapter_config_path):
        # If the checkpoint is trained with LoRA, we load and merge the adapters
        # NOTE: move the model to cuda can greatly speed up merge_and_unload()

        """
        Assumes the model has been trained for K rounds, each with an adapter.
        We will load the adapters sequentially and merge them to the base model.
        If training, we will also save the adapters to the output_dir for loading
        later, where the folder sturcture will be like:
            output_dir/
                adapter_0/
                    adapter_config.json
                    adapter_model.safetensors
                adapter_1/
                    adapter_config.json
                    adapter_model.safetensors
                adapter_config.json (the current adapter config if training)
                adapter_model.safetensors (the adapter to be saved during training)
        """
        adapter_id = 0
        adapter_dir = os.path.join(checkpoint_dir, f"adapter_{adapter_id}")
        while os.path.exists(adapter_dir):
            print(f"Loading and merging adapter {adapter_id} from {adapter_dir}")
            peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
            if copy_adapters_to_new_output_dir:
                save_dir = os.path.join(train_args.output_dir, f"adapter_{adapter_id}")
                peft_model.save_pretrained(save_dir)
            base_model = peft_model.merge_and_unload()
            adapter_id += 1
            adapter_dir = os.path.join(checkpoint_dir, f"adapter_{adapter_id}")

        print(f"Loading and merging adapter {adapter_id} from {checkpoint_dir}")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        if copy_adapters_to_new_output_dir:
            save_dir = os.path.join(train_args.output_dir, f"adapter_{adapter_id}")
            peft_model.save_pretrained(save_dir)
        base_model = peft_model.merge_and_unload()

    if is_training:
        if new_lora_config is not None:
            print(f"Creating a new LoRA adapter: {new_lora_config}")
            model = get_peft_model(base_model, new_lora_config)
            model.print_trainable_parameters()
        else:
            model = base_model
            if train_args.llm_train_mode == "full":
                model.requires_grad_(True)
            else:
                assert train_args.llm_train_mode == "frozen"
                model.requires_grad_(False)
                model.mm_projector.requires_grad_(True)
            print_trainable_parameters(model)
    else:
        model = base_model
        model.eval()
    torch.cuda.empty_cache()

    # build tokenizer
    tokenizer = build_tokenizer_and_update_config(model.config)

    if (
        is_training
        and train_args is not None
        and train_args.output_dir
        and train_args.local_rank in [-1, 0]
    ):
        model.config.save_pretrained(train_args.output_dir)
    return model, tokenizer


if __name__ == "__main__":
    import os
    from transformers import Trainer
    from mmassist.configs import parse_args

    model_args, train_args = parse_args()
    model_args.img_patch_token_size = 3
    train_args.llm_train_mode = "lora"
    output_dir = train_args.output_dir = os.path.join(
        os.path.dirname(__file__), "../../outputs", "debug"
    )
    model, tokenizer = build_from_pretrained(model_args, train_args)

    trainer = Trainer(model=model)
    trainer.save_model(train_args.output_dir)

    model_args.img_patch_token_size = 5
    train_args.llm_train_mode = "full"
    train_args.output_dir = output_dir.replace("debug", "debug_resume")

    model_loaded, tokenizer_loaded = build_from_checkpoint(
        output_dir,
        is_training=True,
        model_args=model_args,
        train_args=train_args,
    )
