# ML Implementations

A temporal museum of machine learning — one self-contained implementation per major milestone, from n-gram language models to modern efficiency techniques.

Each folder is a complete, standalone artifact.

## The Implementations

| # | Folder | Era | Stack | CPU time |
|---|--------|-----|-------|----------|
| 01 | [ngram-lm](01-ngram-lm/) | 1948–1990s | stdlib + NumPy | < 5s |
| 02 | [backprop](02-backprop/) | 1986 | NumPy | < 1s |
| 03 | [word-vectors](03-word-vectors/) | 2013 | NumPy | ~10 min |
| 04 | [backprop-through-time](04-backprop-through-time/) | 1986–1997 | NumPy | ~5 min |
| 05 | [seq2seq-attention](05-seq2seq-attention/) | 2014 | PyTorch | ~10 min |
| 06 | [transformer](06-transformer/) | 2017 | PyTorch | ~15 min |
| 07 | [bert-mlm](07-bert-mlm/) | 2018 | PyTorch | ~20 min |
| 08 | [gpt](08-gpt/) | 2018–2019 | PyTorch | ~15 min |
| 09 | [scaling-laws](09-scaling-laws/) | 2020 | PyTorch | ~60 min |
| 10 | [lora](10-lora/) | 2022 | PyTorch | ~10 min |
| 11 | [rlhf](11-rlhf/) | 2022 | PyTorch + trl | ~30 min |
| 12 | [modern-efficiency](12-modern-efficiency/) | 2021–2022 | PyTorch | ~5 min |

## Setup

```bash
pip install -r requirements.txt

# For RLHF only:
pip install -r 11-rlhf/requirements.txt
```

## Running the Tests

```bash
pytest */test_*.py -v

# The scaling-laws sweep is slow — run it explicitly:
pytest 09-scaling-laws/test_scaling.py -v --run-slow
```

Tests are deterministic (fixed seeds), fast (< 30s per folder), and behavioral — they check that the implementation does the right thing, not just that it runs.

## Design Decisions

**Why NumPy for folders 01–04?** Autograd hides the learning. Seeing `dL/dW1 = a1.T @ delta2` teaches more than `loss.backward()`. The early implementations make the math concrete.

**Why PyTorch from folder 05 onward?** The mechanisms — attention, masking, positional encoding — are the lesson. The training plumbing is not. PyTorch handles the bookkeeping so the code can focus on the ideas.

**Why no shared utilities?** Each folder is complete. No cross-folder imports. This is a deliberate choice: it means each folder is slightly more verbose than it could be, but it also means you never need to understand another folder to understand the one you're reading.

**Why these 12?** They represent the moments where something genuinely changed — a new capability, a new problem solved, a new way of thinking. Each one is a hinge point in the history of the field.
