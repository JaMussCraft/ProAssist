import os
import sys
import time
import argparse
import submitit
from tqdm import tqdm

from transformers import HfArgumentParser

from mmassist.data import build_eval_datasets
from mmassist.model import build_from_checkpoint
from mmassist.configs.arguments import EvalArguments, SlurmArguments
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator


class Task:
    def __call__(
        self, args: EvalArguments, local_rank: int, global_rank: int, num_tasks: int
    ):

        # global rank
        # job_env = submitit.JobEnvironment()
        # local_rank = job_env.local_rank
        # global_rank = job_env.global_rank
        # num_tasks = job_env.num_tasks

        # load model
        args_dict = args.to_dict()
        print(args_dict)
        device = f"cuda:{local_rank}"
        model, tokenizer = build_from_checkpoint(args.model_path, rank=local_rank)

        # build the evaluation datasets
        model_args = model.config.to_dict()
        args_dict.update(model_args)
        datasets = build_eval_datasets(**args_dict, keep_images=False)

        # run the evaluators
        for evaluator_name in args.evaluators_to_run.split(","):
            if evaluator_name == "stream":
                for dn, dataset in datasets.items():
                    evaluator = StreamEvaluator.build(
                        dataset,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        **args_dict,
                    )
                    # get the sample indices for this rank

                    num_samples = len(dataset)
                    sample_indices = list(range(global_rank, num_samples, num_tasks))

                    print(f"[{dn}] [Rank {global_rank}]: {len(sample_indices)} samples")
                    start_time = time.time()
                    all_metrics = evaluator.compute_metrics(sample_indices)
                    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")
            else:
                raise NotImplementedError(
                    f"Evaluator {evaluator_name} is not implemented."
                )

        return all_metrics


def main(eval_args: EvalArguments, slurm_args: SlurmArguments):
    executor = submitit.AutoExecutor(folder=slurm_args.output_dir)

    num_jobs = slurm_args.num_jobs

    executor.update_parameters(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=slurm_args.cpus_per_task,
        slurm_partition=slurm_args.partition,
        name=slurm_args.job_name,
        mem_gb=slurm_args.mem_gb,
        timeout_min=slurm_args.timeout_min,
    )
    if slurm_args.account:
        executor.update_parameters(account=slurm_args.account)

    jobs = []
    with executor.batch():
        for gr in range(num_jobs):
            # lr = gr // 8
            lr = 0
            jobs.append(executor.submit(Task(), eval_args, lr, gr, num_jobs))

    # only to wait for the jobs to finish
    _ = [job.result() for job in jobs]

    # gather results
    args_dict = eval_args.to_dict()
    datasets = build_eval_datasets(**args_dict, keep_images=False)
    # run the evaluators
    for evaluator_name in eval_args.evaluators_to_run.split(","):
        for dn, dataset in datasets.items():
            evaluator = StreamEvaluator.build(dataset, **args_dict)
            scores = evaluator.gather_metrics()
            print(f"{dn} - {evaluator_name}:")
            print(scores)

    return 0


if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments, SlurmArguments))
    eval_args, slurm_args = parser.parse_args_into_dataclasses()
    main(eval_args, slurm_args)
