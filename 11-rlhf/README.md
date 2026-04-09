# 11 — RLHF: Reinforcement Learning from Human Feedback

## Context

Ouyang et al. 2022, "Training language models to follow instructions with human feedback" (InstructGPT), introduced the three-stage pipeline that is now the standard recipe for building assistant-style LLMs:

1. **SFT** — Supervised Fine-Tuning on high-quality (prompt, completion) pairs written by human contractors. Establishes the basic ability to follow instructions.
2. **Reward Model** — Train a separate model to predict which of two completions a human would prefer (using the Bradley-Terry pairwise ranking model). This converts sparse, expensive human judgements into a dense, differentiable signal.
3. **RL Fine-Tuning** — Use the reward model as a proxy for human preference and optimize the policy via reinforcement learning (PPO in the original paper). A KL penalty against the SFT model prevents reward hacking.

Why not just do more SFT? Because SFT can only teach the model to imitate the training distribution. RLHF teaches the model to *optimize* for human preferences, including on inputs it has never seen. The reward model generalizes the collected human judgements to new examples.

---

## What this code does

### `train_sft.py` — Stage 1
Trains a small GPT-style language model on synthetically generated (prompt, completion) pairs of the form `"summarize: the fox slept near the river" → "fox near river"`. Uses character-level tokenization (fully self-contained; no external tokenizer). Saves `sft_model.pt`.

### `train_rm.py` — Stage 2
Constructs pairwise preference data: the ideal completions from Stage 1 are "chosen", and algorithmically degraded versions (reversed words, random padding, repetition) are "rejected". Trains `RewardModel` with Bradley-Terry loss. Prints per-step loss and accuracy (% of pairs where r_chosen > r_rejected). Saves `reward_model.pt`.

### `train_rl.py` — Stage 3
Initializes the policy from the SFT checkpoint, then runs REINFORCE with a KL penalty. For each batch of prompts, the policy generates completions, the reward model scores them, and the KL term penalizes divergence from the original SFT distribution. After 200 steps the `reward_hacking_demo()` function removes the KL penalty and runs 50 more steps — the policy rapidly finds degenerate completions that score high on the reward model but are semantically collapsed (repeated tokens, nonsense strings).

### `reward_model.py`
Self-contained `GPTBase` + `RewardModel` + `bradley_terry_loss`. No dependencies on external libraries beyond PyTorch.

---

## Key implementation details

### Bradley-Terry loss

The loss has a one-line derivation:

```
P(chosen ≻ rejected) = sigmoid(r_chosen - r_rejected)   # Bradley-Terry model
MLE → maximize log P → minimize -log sigmoid(r_chosen - r_rejected)
```

In code: `loss = -F.logsigmoid(r_chosen - r_rejected).mean()`

This is numerically equivalent to binary cross-entropy on the margin, and it is exactly what is implemented in `reward_model.py`.

### KL penalty — the critical safety mechanism

Without the KL penalty, the policy optimizes `reward_model(completion)` directly. Since the reward model is an imperfect proxy, the policy learns to exploit its blind spots — a phenomenon called **reward hacking**. Completions that score high on the reward model may be semantically degenerate.

The full objective is:

```
maximize  E[r(prompt, completion)] − β · KL(π_RL || π_SFT)
```

The KL term keeps the policy close to the SFT distribution, ensuring it retains the properties that made SFT outputs good in the first place. Without it the policy drifts to a degenerate mode within tens of steps (visible in `reward_hacking_demo`).

### REINFORCE vs PPO

This implementation uses REINFORCE (Williams 1992), not PPO. The difference:

| | REINFORCE | PPO |
|---|---|---|
| Gradient estimate | `∑ log π(a) · R` | Clipped ratio `min(r·A, clip(r,1±ε)·A)` |
| Variance | High — needs many samples | Lower — uses value function baseline |
| Stability | Noisier | More stable |
| Complexity | ~20 lines | ~100+ lines + critic network |

REINFORCE is sufficient to demonstrate the concept and show reward hacking. In production (InstructGPT, ChatGPT), PPO is used for its lower variance and better sample efficiency.

---

## What's deliberately omitted

**Full PPO with value function.** A PPO implementation requires a critic (value) network, advantage estimation (GAE), multiple epochs of mini-batch updates per rollout, and careful hyperparameter tuning. The `trl` library (Hugging Face) provides a production-quality PPO trainer if you need that.

**DPO (Direct Preference Optimization, Rafailov et al. 2023).** DPO eliminates the RL loop entirely by deriving a closed-form preference loss directly from the pairwise data. Simpler to implement and often competitive with PPO-based RLHF. No reward model, no sampling loop — just supervised fine-tuning on a modified objective.

**Constitutional AI (Anthropic 2022).** Instead of human preference labels, CAI uses a set of principles and a second LLM to generate critique-revision pairs. The RL stage is replaced by RLAIF (RL from AI Feedback). Requires a capable LLM in the loop; not implementable with toy models.

**Reward model ensembling.** Using multiple reward models and taking the minimum score reduces overoptimization of any single model's weaknesses. Standard in production systems, omitted here for clarity.
