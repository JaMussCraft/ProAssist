import os
import json
import torch
import sentence_transformers as sbert
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from mmassist.eval.runners.stream_inference import FrameOutput


@torch.no_grad()
def get_semantic_match_cost(
    strings: list[tuple[str, str]], model: sbert.SentenceTransformer, **kwargs
) -> torch.Tensor:
    """Calculate the semantic text similarity cost between two lists of strings.

    :param strings: list of tuples of strings to compute similarity between
    :param model: a sentence transformer model for computing sentence embeddings

    :return: tensor of semantic text similarity cost which
        is the negative log of the cosine similarity between the embeddings
    """
    all_strings = [s for pair in strings for s in pair]
    embeddings = model.encode(
        all_strings, convert_to_tensor=True, show_progress_bar=False, **kwargs
    )
    embeddings = embeddings.view(len(strings), 2, -1)
    cos_sim = sbert.util.pairwise_cos_sim(embeddings[:, 0], embeddings[:, 1])
    score = 1 - cos_sim.abs()
    # for i in range(0, len(all_strings), 2):
    #     print(f"{all_strings[i]} -> {all_strings[i+1]}: {1-score[i//2]:.3f}")
    return score


def mask_local_cost(
    cost: torch.Tensor, match_window: tuple[int, int], masked_cost: float = torch.inf
) -> torch.Tensor:
    h, w = cost.shape
    mask = torch.arange(1, h + 1)[:, None] - torch.arange(1, w + 1)
    mask = -mask
    l, r = match_window
    cost[(mask < l) | (mask > r)] = masked_cost
    return cost


def get_text_match_cost(
    eval_outputs: list[FrameOutput],
    sts_model: sbert.SentenceTransformer | None,
    match_window: tuple[int, int],
    no_talk_str: str = "",
    **kwargs,
) -> torch.Tensor:

    gen_talk_times = torch.tensor(
        [-1.0 if o.gen == no_talk_str else 1.0 for o in eval_outputs]
    )
    ref_talk_times = torch.tensor(
        [-1.0 if o.ref == no_talk_str else 1.0 for o in eval_outputs]
    )
    cost = -gen_talk_times[None].T @ ref_talk_times[None]
    cost = (cost + 1) / 2

    if sts_model is not None:
        l, r = match_window  # only compute for nearby frames to save some compute
        cmp_ids = []
        cmp_texts = []
        for i, fi in enumerate(eval_outputs):
            gen_txt = fi.gen
            if gen_txt != no_talk_str:
                for j in range(max(i + l, 0), min(i + r + 1, len(eval_outputs))):
                    ref_txt = eval_outputs[j].ref
                    if ref_txt != no_talk_str:
                        cmp_ids.append((i, j))
                        cmp_texts.append((gen_txt, ref_txt))

        if cmp_texts:
            sem_costs = get_semantic_match_cost(cmp_texts, sts_model, **kwargs)
            for (i, j), c in zip(cmp_ids, sem_costs):
                cost[i, j] = c

    return cost, gen_talk_times


def get_distance_cost(
    h: int, w: int, power: float = 1.5, multiplier: float = 1.0
) -> torch.Tensor:
    dist = torch.range(1, h)[:, None] - torch.range(1, w)
    dist = dist.abs().float()
    return (dist**power) * multiplier


@dataclass
class MatchResult:
    matched: list[tuple[FrameOutput, FrameOutput]]
    missed: list[FrameOutput]
    redundant: list[FrameOutput]
    match_costs: list[float]
    semantic_scores: list[float]

    @classmethod
    def from_json(cls, d: dict) -> "MatchResult":
        return cls(
            matched=[(FrameOutput(**g), FrameOutput(**r)) for g, r in d["matched"]],
            missed=[FrameOutput(**m) for m in d["missed"]],
            redundant=[FrameOutput(**m) for m in d["redundant"]],
            match_costs=d["match_costs"],
            semantic_scores=d["semantic_scores"],
        )

    def to_json(self) -> dict:
        return {
            "matched": [(g.to_dict(), r.to_dict()) for g, r in self.matched],
            "missed": [m.to_dict() for m in self.missed],
            "redundant": [m.to_dict() for m in self.redundant],
            "match_costs": self.match_costs,
            "semantic_scores": self.semantic_scores,
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


def find_match(
    eval_outputs: list[FrameOutput],
    sts_model: sbert.SentenceTransformer | None,
    match_window: tuple[int, int] = [5, 5],
    dist_func_factor: float = 0.2,
    dist_func_power: float = 1.5,
    no_talk_str: str = "",
    debug: bool = False,
    **kwargs,
) -> MatchResult:

    # compute the text match cost
    match_cost, gen_talk_ids = get_text_match_cost(
        eval_outputs,
        sts_model,
        match_window=match_window,
        no_talk_str=no_talk_str,
        **kwargs,
    )

    # compute the distance cost
    rg = max(abs(match_window[0]), match_window[1])
    multiplier = dist_func_factor / (rg**dist_func_power)
    dist_cost = get_distance_cost(
        *match_cost.shape, power=dist_func_power, multiplier=multiplier
    )
    dist_cost = mask_local_cost(dist_cost, match_window)

    # combine the costs
    total_cost = match_cost + dist_cost

    # find the optimal matching using the LAPJVsp algorithm
    gen_talk_pos_mask = gen_talk_ids == 1
    gen_talk_indices = gen_talk_pos_mask.nonzero().flatten()
    gen_to_ref_costs = total_cost[gen_talk_pos_mask]
    gen_to_ref_costs_csr = csr_matrix(gen_to_ref_costs.numpy())
    idx_in_gen_talk, idx_in_ref = min_weight_full_bipartite_matching(
        gen_to_ref_costs_csr
    )
    idx_in_gen = gen_talk_indices[idx_in_gen_talk].numpy()
    gen_to_ref_match = {i: j for i, j in zip(idx_in_gen, idx_in_ref)}
    ref_be_matched = set(idx_in_ref)

    ### debug
    if debug:
        print("match_cost")
        print(match_cost)
        print("dist_cost")
        print(dist_cost)
        print("total_cost")
        print(total_cost)
        for i, j in zip(idx_in_gen, idx_in_ref):
            print(f"gen {i}: {eval_outputs[i].gen}")
            print(f"-> ref {j}: {eval_outputs[j].ref}")
            t, m, d = total_cost[i, j], match_cost[i, j], dist_cost[i, j]
            print(f"   total_cost: {t:.3f}, match_cost: {m:.3f}, dist_cost: {d:.3f}")

    matched, missed, redundant = [], [], []
    match_costs, semantic_scores = [], []
    for i, f in enumerate(eval_outputs):
        if i in gen_to_ref_match:
            ref_frame = eval_outputs[gen_to_ref_match[i]]
            if ref_frame.ref != no_talk_str:
                matched.append((f, ref_frame))
                match_costs.append(total_cost[i, gen_to_ref_match[i]].item())
                semantic_scores.append(1 - match_cost[i, gen_to_ref_match[i]].item())
            else:
                redundant.append(f)
        if f.ref != no_talk_str and i not in ref_be_matched:
            missed.append(f)

    return MatchResult(
        matched=matched,
        missed=missed,
        redundant=redundant,
        match_costs=match_costs,
        semantic_scores=semantic_scores,
    )


if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sts_model = sbert.SentenceTransformer(model_name, device=device)

    test_outputs = [
        FrameOutput(gen="", ref="Hi"),
        *[FrameOutput(gen="", ref="")] * 5,
        FrameOutput(gen="Hello", ref=""),
        FrameOutput(gen="", ref="running"),
        FrameOutput(gen="A boy is running.", ref="A cat is chasing the ball."),
        *[FrameOutput(gen="", ref="")] * 5,
        FrameOutput(gen="", ref="A boy is running fast."),
        FrameOutput(gen="A cat is playing with the ball.", ref=""),
        FrameOutput(gen="Boy picks up a bottle", ref=""),
        FrameOutput(gen="", ref="cat plays with bottle"),
        FrameOutput(gen="A boy is running really fast.", ref=""),
    ]

    match_result = find_match(
        test_outputs,
        sts_model=sts_model,
        match_window=[-4, 2],
        dist_func_factor=0.3,
        dist_func_power=1.5,
        batch_size=64,
        debug=True,
    )

    from pprint import pprint

    # set torch print all
    torch.set_printoptions(profile="full", sci_mode=False, precision=3)
    pprint(match_result.to_json())
