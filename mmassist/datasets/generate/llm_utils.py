import os


class LLMGenerator:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.default_sampling_args = dict(
            n=1, temperature=0.5, top_p=0.95, max_tokens=128 * 1024  # 128K
        )
        global SamplingParams
        from vllm import SamplingParams

    @classmethod
    def build(cls, model_id: str, number_gpus: int, local_rank: int | None = None):
        if local_rank is not None:
            # trick to use vLLM in slurm environment
            gpu_ids = [local_rank * number_gpus + i for i in range(number_gpus)]
            gpus = ",".join(str(g) for g in gpu_ids)
            # print process id
            print(f"Process ID: {os.getpid()}")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM
        from transformers import AutoTokenizer

        llm = LLM(
            model=model_id,
            tensor_parallel_size=number_gpus,
            gpu_memory_utilization=0.96,
            swap_space=0,
            max_num_seqs=64,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(llm, tokenizer)

    def generate(
        self, inputs: list[tuple[str, str]], **vllm_sampling_args
    ) -> list[str]:

        messages = [{"role": r, "content": c} for r, c in inputs]
        prompts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        sampling_args = self.default_sampling_args.copy()
        sampling_args.update(vllm_sampling_args)
        sampling_params = SamplingParams(**sampling_args)
        num_repeats = sampling_args.get("n", 1)

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        texts = [outputs[0].outputs[i].text for i in range(num_repeats)]

        return texts

    def batch_generate(
        self, inputs: list[list[tuple[str, str]]], **vllm_sampling_args
    ) -> list[list[str]]:
        messages = [[{"role": r, "content": c} for r, c in batch] for batch in inputs]
        prompts = [
            self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            for m in messages
        ]

        sampling_args = self.default_sampling_args.copy()
        sampling_args.update(vllm_sampling_args)
        sampling_params = SamplingParams(**sampling_args)
        num_repeats = sampling_args.get("n", 1)

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        texts = [
            [outputs[j].outputs[i].text for i in range(num_repeats)]
            for j in range(len(inputs))
        ]

        return texts
