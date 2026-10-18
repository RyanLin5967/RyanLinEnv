"""Sampler equivalence of the CPU transformers engine with mlx-vlm 0.6.8 (torch only).

The reference below is written from mlx_vlm.sample_utils' definitions with plain Python
loops, independently of local_hf's tensor code.
"""

import math
import random

import pytest

torch = pytest.importorskip("torch")

from vision_rl import local_hf  # noqa: E402

SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "max_tokens": 16384}


def reference(logits, generated, *, temperature, top_p, top_k, presence_penalty, context=20):
    values = list(logits)
    for token in set(generated[-context:]):
        values[token] -= presence_penalty
    norm = math.log(sum(math.exp(v) for v in values))
    logprobs = [v - norm for v in values]
    ascending = sorted(range(len(values)), key=lambda i: logprobs[i])
    running, keep = 0.0, set()
    for index in ascending:
        running += math.exp(logprobs[index])
        if running > 1 - top_p:
            keep.add(index)
    masked = [lp if i in keep else -math.inf for i, lp in enumerate(logprobs)]
    ranked = sorted(range(len(values)), key=lambda i: masked[i], reverse=True)[:top_k]
    return [masked[i] / temperature if i in set(ranked) else -math.inf
            for i in range(len(values))]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_filter_matches_mlx_definition(seed):
    rng = random.Random(seed)
    logits = [rng.gauss(0, 3) for _ in range(64)]
    generated = [rng.randrange(64) for _ in range(30)]
    got = local_hf.mlx_filter(torch.tensor([logits], dtype=torch.float32), generated,
                              temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                              presence_penalty=1.5)[0].tolist()
    want = reference(logits, generated, temperature=0.7, top_p=0.8, top_k=20,
                     presence_penalty=1.5)
    assert [math.isinf(v) for v in got] == [math.isinf(v) for v in want]
    for g, w in zip(got, want):
        if not math.isinf(w):
            assert g == pytest.approx(w, abs=1e-4)


def test_presence_penalty_only_counts_last_twenty_generated_tokens():
    logits = torch.zeros(1, 30)
    out = local_hf.mlx_filter(logits, [5] + list(range(10, 30)), temperature=1.0, top_p=1.0,
                              top_k=30, min_p=0.0, presence_penalty=1.5)[0]
    assert out[5] > out[10]  # token 5 is 21 tokens back: not penalised
    assert out[10] == pytest.approx(out[29])


def test_min_p_is_refused():
    with pytest.raises(ValueError, match="min_p"):
        local_hf.mlx_filter(torch.zeros(1, 4), [], temperature=1.0, top_p=1.0, top_k=2,
                            min_p=0.1, presence_penalty=0.0)


def test_sampler_forces_its_draw_and_is_seeded():
    prompt = torch.tensor([[7, 8, 9]])
    scores = torch.randn(1, 50, generator=torch.Generator().manual_seed(3))
    first = local_hf.MLXEquivalentSampler(SAMPLING, 1234, 3)
    forced = first(prompt, scores.clone())
    token = first.drawn[0]
    assert forced[0, token] == 0 and torch.isinf(forced).sum() == 49
    second = local_hf.MLXEquivalentSampler(SAMPLING, 1234, 3)
    second(prompt, scores.clone())
    assert second.drawn == [token]
    follow = torch.cat([prompt, torch.tensor([[token]])], dim=1)
    first(follow, scores.clone())
    assert len(first.drawn) == 2
    wrong = torch.cat([prompt, torch.tensor([[(token + 1) % 50]])], dim=1)
    with pytest.raises(ValueError, match="did not draw"):
        second(wrong, scores.clone())


def test_draws_follow_the_filtered_distribution():
    scores = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0]])
    filtered = local_hf.mlx_filter(scores, [], temperature=0.7, top_p=0.8, top_k=20,
                                   min_p=0.0, presence_penalty=1.5)
    expected = torch.softmax(filtered, dim=-1)[0]
    counts = torch.zeros(5)
    for seed in range(4000):
        sampler = local_hf.MLXEquivalentSampler(SAMPLING, seed, 0)
        sampler(torch.zeros(1, 0, dtype=torch.long), scores.clone())
        counts[sampler.drawn[0]] += 1
    assert counts[3] == counts[4] == 0  # outside top-p
    assert (counts / 4000 - expected).abs().max() < 0.03
