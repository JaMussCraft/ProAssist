import time
import submitit
from pprint import pprint

from transformers import HfArgumentParser

from mmassist.data import build_eval_datasets
from mmassist.model import build_from_checkpoint
from mmassist.model.configuration_proact import ProActLlamaConfig
from mmassist.configs.arguments import EvalArguments, SlurmArguments
from mmassist.eval.evaluators import evaluator_name_to_cls
from mmassist.eval.eval_utils import parse_inference_setups


MODE = [
    "slurm_inference",
    "local_inference",
    "local_mp_inference",
    "compute_metrics",
    "llm_eval",
]


def run_eval(
    args: EvalArguments,
    mode: str = "slurm_inference",
    verbose: bool = False,
    rank=None,
    num_tasks=None,
    llm=None,
    **kwargs,
):
    """Run evaluation on the given model and inference setups

    :param args: EvalArguments, the arguments for evaluation
    :param mode: one of MODE
    :param verbose: whether to print verbose information
    :param rank: the rank of the process. Only used in local_mp_inference mode
    :param num_tasks: the number of tasks. Only used in local_mp_inference mode
    :param llm: the LLM generator. Only used in llm_eval
    """
    args_dict = args.to_dict()
    assert mode in MODE, f"Invalid mode: {mode}; must be one of {MODE}"
    llm = llm
    if "inference" in mode:
        if mode == "slurm_inference":
            job_env = submitit.JobEnvironment()
            local_rank = job_env.local_rank
            global_rank = job_env.global_rank
            num_tasks = job_env.num_tasks
            print("Job ID: ", job_env.job_id)
        elif mode == "local_inference":
            local_rank = global_rank = 0
            num_tasks = 1
        elif mode == "local_mp_inference":
            local_rank = global_rank = rank
        else:
            raise ValueError(f"Invalid mode: {mode}")

        assert local_rank is not None, "rank must be provided"
        assert num_tasks is not None, "num_tasks must be provided"

        # load model
        model, tokenizer = build_from_checkpoint(args.model_path, rank=local_rank)
        model_config = model.config

    else:
        local_rank, global_rank, num_tasks = None, None, None
        model, tokenizer = None, None
        model_config = ProActLlamaConfig.from_pretrained(args.model_path)
        if mode == "llm_eval":
            assert llm is not None, "llm must be provided"

    # dataset build args
    model_args = model_config.to_dict()
    args_dict.update(model_args)
    args_dict["print_info"] = True

    # run the evaluators one by one
    inference_setups = parse_inference_setups(args.inference_setups)
    # print(args_dict)
    if verbose:
        print("Model: ", args.model_path)
        pprint(inference_setups)
    for data_name, evaluator_and_setups in inference_setups.items():
        datasets = build_eval_datasets(eval_datasets=data_name, **args_dict)
        dataset = list(datasets.values())[0]
        dataset.remove_summarize_turns = False if "summary" in data_name else True
        for evaluator_name, setups in evaluator_and_setups.items():
            evaluator_cls = evaluator_name_to_cls[evaluator_name]
            evaluator = evaluator_cls.build(
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                device=model.device if model is not None else None,
                **args_dict,
            )
            for setup in setups:
                evaluator.update_eval_setup(**setup)
                if "context_handling_method" in setup:
                    ctx_method = setup["context_handling_method"]
                    model_config.exceed_context_handling = ctx_method
                    args_dict["exceed_context_handling"] = ctx_method
                args_dict["eval_setup"] = setup
                evaluator.save_args(args_dict)
                log_str = f"Evalulation: {evaluator.eval_name}"
                if "inference" in mode:
                    # if in inference mode, run inference and compute individual metrics

                    sample_indices = list(range(global_rank, len(dataset), num_tasks))
                    num_rank = len(sample_indices)
                    log_str += f"\n[Rank {global_rank}] processing {num_rank} samples"
                    if verbose:
                        print(log_str)
                    if global_rank in [7, 8] and "ego4d" in dataset:
                        print("Skipping ego4d for rank ", global_rank)
                        continue
                    evaluator.run_all_predictions(sample_indices, progress_bar=verbose)
                elif mode == "compute_metrics":
                    # gather metrics
                    scores = evaluator.compute_metrics(**kwargs)
                    if verbose:
                        log_str += f"\nMetrics:\n"
                        for metric, score in scores.items():
                            log_str += f"{metric}: {score:.4f}\n"
                        print(log_str)
                elif mode == "llm_eval":
                    # run llm evaluation
                    runner = setup["inference_runner_type"]
                    assert runner == "stream", "LLM eval is only for stream runner"
                    evaluator.llm_eval(llm=llm, num_repeat=args.number_repeat, **kwargs)

    # no need to return anything as we should have saved all we need :-)


def main(eval_args: EvalArguments, slurm_args: SlurmArguments):
    start_time = time.time()
    executor = submitit.AutoExecutor(folder=slurm_args.log_dir)
    executor.update_parameters(
        name=slurm_args.job_name,
        nodes=slurm_args.num_nodes,
        tasks_per_node=slurm_args.tasks_per_node,
        gpus_per_node=slurm_args.gpus_per_node,
        cpus_per_task=slurm_args.cpus_per_node // slurm_args.tasks_per_node,
        slurm_partition=slurm_args.partition,
        mem_gb=slurm_args.mem_gb,
        timeout_min=slurm_args.timeout_min,
    )
    if slurm_args.account:
        executor.update_parameters(slurm_account=slurm_args.account)
    if slurm_args.slurm_exclude:
        executor.update_parameters(slurm_exclude=slurm_args.slurm_exclude)
    job = executor.submit(run_eval, eval_args, "slurm_inference", verbose=True)
    job.results()  # wait for the job to finish
    run_eval(eval_args, mode="compute_metrics", verbose=True)
    print(f"All Finished! Time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Model: {eval_args.model_path}")
    runs = "\n".join(eval_args.inference_setups.split(","))
    print(f"Runs:\n{runs}")


if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments, SlurmArguments))
    eval_args, slurm_args = parser.parse_args_into_dataclasses()
    main(eval_args, slurm_args)
