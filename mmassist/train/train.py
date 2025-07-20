import os

os.environ["WANDB__SERVICE_WAIT"] = "600"

from mmassist.configs import parse_args
from mmassist.data import build_train_dataset, build_eval_datasets, ProActCollator
from mmassist.model import build_from_pretrained, build_from_checkpoint
from mmassist.train.utils import is_global_rank_zero
from mmassist.train.trainer import CustomTrainer


# def compute_metrics(eval_predictions: EvalPrediction, fps: int = 2, **kwargs):
#     lm_ppl, frame_diff, fluency, lm_correctness = (
#         torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
#     )
#     return {
#         f"lm_ppl": lm_ppl,
#         f"time_diff": frame_diff / fps,
#         f"fluency": fluency,
#         f"lm_correctness": lm_correctness,
#     }


def train():
    model_args, train_args = parse_args()
    all_args_dict = {**model_args.to_dict(), **train_args.to_dict()}
    if is_global_rank_zero():
        exp_setup = "\n".join([f"{k}: {v}" for k, v in all_args_dict.items()])
        print(f"{'*'*40} Experiment setup {'*'*40}:\n{exp_setup}\n{'*'*100}")

    if not train_args.resume_from_checkpoint:
        model, tokenizer = build_from_pretrained(model_args, train_args)
    else:
        model, tokenizer = build_from_checkpoint(
            train_args.resume_from_checkpoint,
            is_training=True,
            model_args=model_args,
            train_args=train_args,
            rank=train_args.local_rank,
        )
    chat_formatter = tokenizer.chat_formatter

    all_args_dict["print_info"] = is_global_rank_zero()
    train_dataset = build_train_dataset(**all_args_dict)
    eval_datasets = build_eval_datasets(**all_args_dict)

    pose_maxlen = 0
    if train_args.use_pose:
        max_seq_len = model_args.max_seq_len
        if model.config.max_position_embeddings > max_seq_len:
            pose_maxlen = model.config.max_position_embeddings
            print(f"Use PoSE to extend position ids: {max_seq_len} -> {pose_maxlen}")
        else:
            train_args.use_pose = False
            print("Set use_pose to False since max_position_embeddings <= max_seq_len")
    data_collator = ProActCollator(
        tokenizer,
        chat_formatter,
        pose_maxlen=pose_maxlen,
        use_binary_decision_head=model_args.use_binary_decision_head,
    )

    train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    train_args.remove_unused_columns = False
    train_args.ddp_find_unused_parameters = False

    if "wandb" in train_args.report_to and is_global_rank_zero():
        import wandb
        import uuid

        # TODO： remove hard-coded project and team name
        wandb.init(
            project="ProAct",
            id=str(uuid.uuid4()),
            name=train_args.run_name,
            group=train_args.run_name,
            entity="meta-intern-yichi",
        )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )
    trainer.train()
    if trainer.args.should_save:
        print("Training finished!")
        trainer.save_model()

    # if eval_datasets is not None:
    #     metrics = {}
    #     for eval_dataset_name, eval_dataset in eval_datasets.items():
    #         trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
    #         metrics.update(
    #             trainer.evaluate(
    #                 eval_dataset=eval_dataset,
    #                 metric_key_prefix=f"eval_{eval_dataset_name}",
    #             )
    #         )
    #     print(metrics)


if __name__ == "__main__":
    train()
