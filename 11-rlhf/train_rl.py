"""
Stage 3: RL Fine-Tuning with Policy Gradient

Use REINFORCE (not full PPO — simpler and sufficient to demonstrate the concept)
with a KL penalty against the SFT model.

The objective:
    reward_with_kl = reward - kl_coeff * KL(policy || sft_policy)

Key insight: without the KL penalty, the policy will find degenerate completions
that score high on the reward model but don't actually make sense.
This is reward hacking, and the KL penalty prevents it.

Run: python train_rl.py   (requires sft_model.pt and reward_model.pt)
"""

import random
import copy
import torch
import torch.nn.functional as F
from reward_model import RewardModel
from train_sft import SFTModel, CharTokenizer, make_sft_data, build_tokenizer

KL_COEFF = 0.1  # how much to penalize divergence from SFT policy
MAX_NEW_TOKENS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def compute_kl(policy_logits, sft_logits):
    """
    KL divergence between policy and SFT model distributions.
    KL(P || Q) = sum P * log(P/Q)

    policy_logits: (B, T, V) or (B, V)
    sft_logits:    same shape

    Returns a scalar: mean KL over all positions and batch elements.
    """
    policy_probs = F.softmax(policy_logits, dim=-1)
    sft_probs = F.softmax(sft_logits, dim=-1)
    # Add eps for numerical safety
    kl = (policy_probs * (policy_probs.log() - sft_probs.log())).sum(dim=-1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Token generation with log-prob tracking
# ---------------------------------------------------------------------------

def generate_with_logprobs(policy, prompt_ids, max_new_tokens, temperature=1.0):
    """
    Generate tokens from the policy, tracking log-probabilities of each
    chosen token (needed for REINFORCE gradient estimate).

    Returns:
        full_ids:  (T_prompt + T_completion,) tensor
        log_probs: list of scalar tensors, one per generated token
    """
    ids = prompt_ids.clone()
    log_probs = []

    for _ in range(max_new_tokens):
        logits = policy(ids.unsqueeze(0))  # (1, T, V)
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,)
        log_prob = torch.log(probs[next_id] + 1e-10)
        log_probs.append(log_prob.squeeze())
        ids = torch.cat([ids, next_id])
        if next_id.item() == 2:  # EOS
            break

    return ids, log_probs


# ---------------------------------------------------------------------------
# Policy gradient step
# ---------------------------------------------------------------------------

def policy_gradient_step(policy, sft_model, reward_model, prompts, tokenizer, optimizer,
                          use_kl=True, temperature=1.0):
    """
    One REINFORCE step:
    1. Generate completions from policy
    2. Score with reward model
    3. Compute KL penalty against SFT model
    4. Loss = -(reward - kl_coeff * kl)  [maximize reward, minimize KL]
    5. Backward + step

    Returns: (mean_reward, mean_kl, mean_reward_with_kl)
    """
    rewards = []
    kl_terms = []
    policy_log_probs = []

    for prompt in prompts:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=DEVICE
        )
        # Generate completion
        full_ids, lp_list = generate_with_logprobs(
            policy, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature
        )

        # Reward from reward model
        rm_input = full_ids.unsqueeze(0)
        if rm_input.shape[1] > 64:
            rm_input = rm_input[:, :64]
        with torch.no_grad():
            reward = reward_model(rm_input).squeeze()

        # KL: compare policy vs SFT on the same token sequence (minus last token)
        if use_kl and len(full_ids) > 1:
            ctx = full_ids[:-1].unsqueeze(0)  # (1, T-1)
            with torch.no_grad():
                sft_logits = sft_model(ctx)   # (1, T-1, V)
            policy_logits = policy(ctx)        # (1, T-1, V)
            kl = compute_kl(policy_logits, sft_logits.detach())
        else:
            kl = torch.tensor(0.0, device=DEVICE)

        rewards.append(reward)
        kl_terms.append(kl)

        # Sum of log-probs of generated tokens (REINFORCE)
        if lp_list:
            policy_log_probs.append(torch.stack(lp_list).sum())
        else:
            policy_log_probs.append(torch.tensor(0.0, device=DEVICE))

    rewards_t = torch.stack(rewards)
    kl_t = torch.stack(kl_terms)
    lp_t = torch.stack(policy_log_probs)

    # Reward adjusted by KL penalty
    adjusted_reward = rewards_t - KL_COEFF * kl_t

    # REINFORCE: loss = -E[adjusted_reward * log_prob]
    # Detach adjusted_reward so gradients flow only through log_probs
    loss = -(adjusted_reward.detach() * lp_t).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return (
        rewards_t.mean().item(),
        kl_t.mean().item(),
        adjusted_reward.mean().item(),
    )


# ---------------------------------------------------------------------------
# Reward hacking demo
# ---------------------------------------------------------------------------

def reward_hacking_demo(policy, reward_model, prompts, tokenizer, n_steps=50):
    """
    Run n_steps WITHOUT KL penalty.
    The policy finds degenerate completions that fool the reward model.
    Print examples of degenerate outputs vs good pre-hack outputs.
    """
    print("\n--- Reward Hacking Demo (no KL penalty) ---")

    # Record some pre-hack samples
    pre_hack_samples = []
    for prompt in prompts[:3]:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=DEVICE
        )
        with torch.no_grad():
            full_ids, _ = generate_with_logprobs(policy, prompt_ids, MAX_NEW_TOKENS)
        text = tokenizer.decode(full_ids.tolist())
        pre_hack_samples.append((prompt, text))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4)

    rewards = []
    for step in range(1, n_steps + 1):
        reward_mean, kl_mean, _ = policy_gradient_step(
            policy, None, reward_model, prompts, tokenizer, optimizer,
            use_kl=False, temperature=0.5,
        )
        rewards.append(reward_mean)
        if step % 10 == 0:
            print(f"  [no-KL] step {step:3d} | reward {reward_mean:.4f}")

    # Post-hack samples
    print("\nPre-hack vs Post-hack completions:")
    for i, (prompt, pre_text) in enumerate(pre_hack_samples):
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=DEVICE
        )
        with torch.no_grad():
            full_ids, _ = generate_with_logprobs(policy, prompt_ids, MAX_NEW_TOKENS)
        post_text = tokenizer.decode(full_ids.tolist())
        print(f"\n  Prompt:    {prompt!r}")
        print(f"  Pre-hack:  {pre_text!r}")
        print(f"  Post-hack: {post_text!r}  <-- degenerate")

    reward_delta = rewards[-1] - rewards[0]
    print(f"\nReward delta after {n_steps} no-KL steps: {reward_delta:+.4f}")
    print("(Large positive delta = policy found reward-hacking shortcuts)")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_rl(n_steps=200, batch_size=8, lr=3e-4, device=None):
    if device is None:
        device = DEVICE

    sft_data = make_sft_data(n=80)
    tokenizer = build_tokenizer(sft_data)
    prompts = [p for p, _ in sft_data]

    # Load SFT model
    ckpt = torch.load("sft_model.pt", map_location=device)
    sft_model = SFTModel(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        num_layers=ckpt["num_layers"],
        max_len=ckpt["max_len"],
    ).to(device)
    sft_model.load_state_dict(ckpt["model_state"])
    sft_model.eval()
    for p in sft_model.parameters():
        p.requires_grad_(False)

    # Policy starts as a copy of SFT model
    policy = copy.deepcopy(sft_model)
    for p in policy.parameters():
        p.requires_grad_(True)
    policy.train()

    # Load reward model
    rm_ckpt = torch.load("reward_model.pt", map_location=device)
    rm = RewardModel(
        vocab_size=rm_ckpt["vocab_size"],
        d_model=rm_ckpt["d_model"],
        num_heads=rm_ckpt["num_heads"],
        num_layers=rm_ckpt["num_layers"],
        max_len=rm_ckpt["max_len"],
    ).to(device)
    rm.load_state_dict(rm_ckpt["model_state"])
    rm.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    print(f"RL fine-tuning for {n_steps} steps (KL coeff={KL_COEFF})...")
    for step in range(1, n_steps + 1):
        batch_prompts = random.sample(prompts, min(batch_size, len(prompts)))
        reward_mean, kl_mean, adj_reward = policy_gradient_step(
            policy, sft_model, rm, batch_prompts, tokenizer, optimizer,
            use_kl=True, temperature=1.0,
        )
        if step % 50 == 0:
            print(
                f"  step {step:4d} | reward {reward_mean:.4f} "
                f"| kl {kl_mean:.4f} | reward_with_kl {adj_reward:.4f}"
            )

    print(f"\nAfter {n_steps} steps with KL penalty: reward_with_kl stable.")

    # Run reward hacking demo
    policy_copy = copy.deepcopy(policy)
    reward_hacking_demo(policy_copy, rm, prompts[:batch_size], tokenizer, n_steps=50)


if __name__ == "__main__":
    train_rl()
