import os
import json
import time
import submitit
from dataclasses import dataclass
from tqdm import tqdm

from llm_utils import LLMGenerator
from dialog_simulation import generate_from_annotation, ParsedVideoAnns
from auto_eval import auto_eval_generated_conversations


@dataclass
class SlurmArguments:
    job_name: str
    num_nodes: int = 1
    tasks_per_node: int = 8
    gpus_per_node: int = 8  # H100
    cpus_per_node: int = 192  # H100
    mem_gb: int = 1800  # per node
    timeout_min: int = 1440
    partition: str = "q1"
    account: str = ""
    log_dir: str = "/data/home/imzyc/project/proactive-assist/slurm_logs/%j"


@dataclass
class PreprocessArgs:
    data_dir: str = "the path to the data directory"
    splits: str = "splits to process, separated by comma"
    output_dir: str = "the path to the output directory"
    llm: str = "the llm to use for generation"
    user_types: str = (
        "user_type@num_gen seperated by comma, e.g. 'no_talk@1,talk_some@2,talk_more@2'",
        "user_type must be one of 'no_talk', 'talk_some', 'talk_more'",
    )
    num_repeats: int = 10
    force_rerun: bool = False
    min_ann_ratio: float = 0.5
    filter_by_llm: bool = False


def run_jobs(
    args: PreprocessArgs,
    anns_per_split: dict[str, list[dict]],
    ann_parse_func: callable,
):
    # get env variables
    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    global_rank = job_env.global_rank
    num_tasks = job_env.num_tasks
    print(f"Rank {global_rank}/{num_tasks}")

    splits = args.splits.split(",")

    # get the samples to run/load for each split for this rank
    anns_to_run_per_split = {}
    anns_to_load_per_split = {}
    for split in splits:
        # get the anns in the split
        all_anns_in_split = anns_per_split[split]

        # filter out the anns that already been processed
        all_anns_to_run, all_anns_to_load = [], []
        for ann in all_anns_in_split:
            vid = ann["video_uid"]  # NOTE: the raw annotation should have this key
            output_file = os.path.join(args.output_dir, split, f"{vid}.json")
            if args.force_rerun:
                all_anns_to_run.append(ann)
            else:
                try:
                    json.load(open(output_file, "r"))  # not found or corrupted
                    all_anns_to_load.append(ann)
                except:
                    all_anns_to_run.append(ann)

        # gather the anns to run
        anns_to_run = all_anns_to_run[global_rank::num_tasks]
        anns_to_load = all_anns_to_load[global_rank::num_tasks]
        print(f"{split}: {len(anns_to_run)} files left:")
        for ann in anns_to_run:
            print("  " + ann["video_uid"])
        anns_to_run_per_split[split] = anns_to_run
        anns_to_load_per_split[split] = anns_to_load

    # parse user types
    user_types = []
    for user_type_with_rep in args.user_types.split(","):
        user_type, num_repeats = user_type_with_rep.split("@")
        user_types.extend([user_type] * int(num_repeats))

    llm, gen_args = None, None
    split_outputs = {}
    for split in args.splits.split(","):
        if split not in split_outputs:
            split_outputs[split] = []

        # process the samples
        for ann in tqdm(anns_to_run_per_split[split]):
            if llm is None:
                # build llm
                llm = LLMGenerator.build(
                    model_id=args.llm,
                    number_gpus=args.tensor_parallel_size,
                    local_rank=local_rank,
                )
                gen_args = {
                    "llm": args.llm,
                    "user_types": args.user_types,
                    "num_repeats": args.num_repeats,
                    "sampling_params": llm.default_sampling_args,
                }

            # parse the annotation
            parsed_ann: ParsedVideoAnns | None = ann_parse_func(ann)
            if parsed_ann is None:
                continue

            # generate the outputs
            outputs = generate_from_annotation(
                parsed_ann,
                llm,
                user_types=user_types,
                num_repeats=args.num_repeats,
                min_ann_ratio=args.min_ann_ratio,
                filter_by_llm=args.filter_by_llm,
            )

            # save the output
            output_dir = os.path.join(args.output_dir, split)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{parsed_ann.video_uid}.json")
            if isinstance(outputs, str):
                with open(output_file, "w") as f:
                    json.dump({"reason_to_exclude": outputs}, f, indent=2)
            else:
                outputs = outputs.to_dict()
                outputs["gen_args"] = gen_args
                with open(output_file, "w") as f:
                    json.dump(outputs, f, indent=2)

            if "reason_to_exclude" not in outputs:
                split_outputs[split].append(outputs)

        # also load the samples that have been processed before
        for ann in anns_to_load_per_split[split]:
            # load the outputs
            vid = ann["video_uid"] if "video_uid" in ann else ann["video_name"]
            output_file = os.path.join(args.output_dir, split, f"{vid}.json")
            outputs = json.load(open(output_file, "r"))
            if "reason_to_exclude" not in outputs:
                split_outputs[split].append(outputs)

    return split_outputs


def main(args: PreprocessArgs, slurm_args: SlurmArguments):
    executor = get_slurm_executor(slurm_args)
    job = executor.submit(run_jobs, args)

    # gather results
    start_time = time.time()
    split_outputs_all_tasks = job.results()
    for split in args.splits.split(","):
        split_outputs = []
        for task_outputs in split_outputs_all_tasks:
            split_outputs.extend(task_outputs.get(split, []))

        # auto-evaluate all the generated dialogs
        for idx, output in enumerate(split_outputs):
            split_outputs[idx] = auto_eval_generated_conversations(output)

        # save the outputs
        print(f"Saving `{split}` split of {len(split_outputs)} videos")
        with open(os.path.join(args.output_dir, f"{split}.json"), "w") as f:
            json.dump(split_outputs, f, indent=2)

    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")


def get_slurm_executor(slurm_args: SlurmArguments) -> submitit.AutoExecutor:
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
    return executor


def save_results(split_outputs_all_tasks: list[dict], splits: str, output_dir: str):
    for split in splits.split(","):
        split_outputs = []
        for task_outputs in split_outputs_all_tasks:
            split_outputs.extend(task_outputs.get(split, []))

        # save the outputs
        print(f"Saving `{split}` split of {len(split_outputs)} videos")
        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(split_outputs, f, indent=2)
