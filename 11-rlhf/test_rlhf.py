"""
Tests for the RLHF implementation.
Run: python -m pytest test_rlhf.py -v
"""

import torch
import torch.nn.functional as F
import pytest
from reward_model import RewardModel, bradley_terry_loss
from train_rl import compute_kl
from train_rm import make_preference_pairs, make_rm_batch, make_worse_completion
from train_sft import make_sft_data, build_tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_rm():
    """A tiny RewardModel for shape/API tests."""
    return RewardModel(vocab_size=50, d_model=16, num_heads=2, num_layers=1, max_len=32)


@pytest.fixture(scope="module")
def tokenizer_and_data():
    sft_data = make_sft_data(n=20)
    tok = build_tokenizer(sft_data)
    return tok, sft_data


# ---------------------------------------------------------------------------
# Reward model output shape
# ---------------------------------------------------------------------------

def test_reward_model_output_shape(small_rm):
    """RewardModel returns shape (batch,) — one scalar per sequence."""
    batch = 4
    seq_len = 16
    token_ids = torch.randint(3, 50, (batch, seq_len))
    rewards = small_rm(token_ids)
    assert rewards.shape == (batch,), f"Expected ({batch},), got {rewards.shape}"


def test_reward_model_output_is_scalar_per_seq(small_rm):
    """Each element of the reward is a single float (no extra dims)."""
    token_ids = torch.randint(3, 50, (2, 8))
    rewards = small_rm(token_ids)
    assert rewards.dim() == 1


# ---------------------------------------------------------------------------
# Bradley-Terry loss
# ---------------------------------------------------------------------------

def test_bradley_terry_loss_direction():
    """Loss is lower when r_chosen > r_rejected than when reversed."""
    r_chosen = torch.tensor([2.0, 1.5, 3.0])
    r_rejected = torch.tensor([0.5, 0.5, 1.0])
    loss_correct = bradley_terry_loss(r_chosen, r_rejected)
    loss_flipped = bradley_terry_loss(r_rejected, r_chosen)
    assert loss_correct < loss_flipped, (
        f"Expected loss_correct ({loss_correct:.4f}) < loss_flipped ({loss_flipped:.4f})"
    )


def test_bradley_terry_loss_is_positive():
    """Bradley-Terry loss is always non-negative."""
    for _ in range(5):
        r_c = torch.randn(8)
        r_r = torch.randn(8)
        loss = bradley_terry_loss(r_c, r_r)
        assert loss.item() >= 0.0


def test_bradley_terry_loss_at_equal_rewards():
    """When r_chosen == r_rejected, loss should equal log(2) ≈ 0.693."""
    r = torch.zeros(4)
    loss = bradley_terry_loss(r, r)
    expected = torch.log(torch.tensor(2.0))
    assert abs(loss.item() - expected.item()) < 1e-5


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def test_kl_is_zero_for_identical_distributions():
    """KL(P || P) = 0."""
    logits = torch.randn(2, 10, 30)
    kl = compute_kl(logits, logits)
    assert abs(kl.item()) < 1e-5, f"KL should be ~0 for identical dists, got {kl.item()}"


def test_kl_is_positive_for_different_distributions():
    """KL(P || Q) > 0 when P != Q."""
    torch.manual_seed(0)
    logits_p = torch.randn(2, 10, 30)
    logits_q = torch.randn(2, 10, 30)
    kl = compute_kl(logits_p, logits_q)
    assert kl.item() > 0.0, f"KL should be positive for different dists, got {kl.item()}"


def test_kl_is_not_symmetric():
    """KL divergence is not symmetric: KL(P||Q) != KL(Q||P) in general."""
    torch.manual_seed(1)
    p = torch.randn(2, 5, 20)
    q = torch.randn(2, 5, 20)
    kl_pq = compute_kl(p, q)
    kl_qp = compute_kl(q, p)
    # They should differ (in general; there's a negligible chance they're equal)
    assert abs(kl_pq.item() - kl_qp.item()) > 1e-6


# ---------------------------------------------------------------------------
# Reward model accuracy after training
# ---------------------------------------------------------------------------

def test_reward_model_scores_differ(tokenizer_and_data):
    """
    After training a small reward model, r_chosen > r_rejected on >60% of pairs.

    This trains a tiny model for a quick sanity-check; not full training.

    Note: max_len=96 is required here. The prompts are ~40 tokens long, so
    truncating to 32 removes all completion tokens, making chosen and rejected
    identical from the model's perspective.
    """
    tokenizer, sft_data = tokenizer_and_data
    triples = make_preference_pairs(sft_data)
    MAX_LEN = 96

    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        max_len=MAX_LEN,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    import random
    random.seed(0)
    torch.manual_seed(0)
    for _ in range(200):
        batch = random.sample(triples, min(8, len(triples)))
        chosen_ids, rejected_ids = make_rm_batch(batch, tokenizer, torch.device("cpu"), max_len=MAX_LEN)
        r_c = model(chosen_ids)
        r_r = model(rejected_ids)
        loss = bradley_terry_loss(r_c, r_r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for triple in triples:
            chosen_ids, rejected_ids = make_rm_batch([triple], tokenizer, torch.device("cpu"), max_len=MAX_LEN)
            r_c = model(chosen_ids)
            r_r = model(rejected_ids)
            if r_c.item() > r_r.item():
                correct += 1
    accuracy = correct / len(triples)
    assert accuracy > 0.60, f"Expected >60% accuracy, got {accuracy:.2%}"


# ---------------------------------------------------------------------------
# Preference pair construction
# ---------------------------------------------------------------------------

def test_make_preference_pairs_length(tokenizer_and_data):
    """make_preference_pairs returns one triple per SFT example."""
    _, sft_data = tokenizer_and_data
    triples = make_preference_pairs(sft_data)
    assert len(triples) == len(sft_data)


def test_preference_pairs_structure(tokenizer_and_data):
    """Each triple is (prompt, chosen, rejected) and rejected != chosen."""
    _, sft_data = tokenizer_and_data
    triples = make_preference_pairs(sft_data)
    for prompt, chosen, rejected in triples:
        assert isinstance(prompt, str)
        assert isinstance(chosen, str)
        assert isinstance(rejected, str)
        assert chosen != rejected, "Chosen and rejected should differ"
