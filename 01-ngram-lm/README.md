# N-gram Language Model

## Context

Claude Shannon's 1948 paper *A Mathematical Theory of Communication* introduced entropy as a measure of information, and proposed modelling language as a stochastic process where the probability of each symbol depends on preceding symbols. N-gram models formalised this: a unigram model assumes each word is drawn independently; a bigram model conditions on the previous word; a trigram model conditions on the previous two.

Through the 1980s and 1990s, trigram language models were the dominant technique in speech recognition and machine translation. They were fast, required no GPU, and scaled to hundreds of millions of tokens on disk. IBM's Candide MT system, CMU Sphinx, and virtually every voice assistant of the era were built on them.

The sparsity problem is fundamental. A trigram model on a 50,000-word vocabulary has 50,000³ = 125 trillion possible contexts. No corpus covers more than a fraction of them, so a naive model assigns zero probability to most trigrams. Smoothing methods — Good-Turing, Katz backoff, and eventually Kneser-Ney (1995) — were developed specifically to redistribute mass to unseen events. Kneser-Ney remains the best-performing count-based smoothing method and is still used as a baseline in LLM evaluation papers.

Neural language models (Bengio et al., 2003) bypassed sparsity by representing words as dense vectors and generalising across similar contexts, which is why they eventually displaced n-gram models for most tasks.

---

## What this code does

- `ngram_lm.py` — Core library:
  - `Tokenizer`: whitespace split, lowercase, strip punctuation, add `<BOS>`/`<EOS>` per line, build vocabulary incrementally.
  - `NgramLM.train()`: iterate over lines, accumulate unigram, bigram, and trigram counts; compute Kneser-Ney continuation counts during training.
  - `NgramLM.laplace_prob()`: Laplace (add-one) smoothed probability with bigram fallback.
  - `NgramLM.kneser_ney_prob()`: Interpolated Kneser-Ney with discount `d=0.75`, three levels of interpolation (unigram KN → bigram KN → trigram KN).
  - `NgramLM.perplexity()`: log-space accumulation, returns `exp(-mean log P)`.
  - `NgramLM.generate()`: temperature-scaled softmax sampling over the vocabulary.

- `demo.py`: trains on an embedded Shakespeare corpus (~400 words), compares perplexity of both smoothing methods on a held-out sentence, generates five sentences from "to be", then deliberately evaluates a sentence containing the OOV word `zylquorx` to show the failure mode.

- `test_ngram.py`: four pytest tests covering normalisation, in-vs-out-of-distribution perplexity, prefix correctness, and the KN ≤ Laplace inequality.

Run the demo:
```
python demo.py
```

Run the tests:
```
pytest test_ngram.py -v
```

---

## Key implementation details

**Kneser-Ney continuation probability.** The unigram component is not raw frequency but a *continuation count*: how many unique left-contexts does word `w` appear in? This is computed during training by storing `bigram_continuation[w] = set(left_contexts)`. The intuition is that a word like "Francisco" appears frequently but almost always after "San", giving it a low continuation count and therefore a small probability of completing a new, unseen context.

**Three-level interpolation.** `kneser_ney_prob` computes P_KN in three stages: (1) KN unigram using continuation counts, (2) KN bigram that discounts the raw bigram count by `d` and adds `lambda * P_KN_unigram`, (3) KN trigram that discounts the raw trigram count and adds `lambda * P_KN_bigram`. Each lambda (interpolation weight) is proportional to the number of distinct right-continuations of the context, ensuring probabilities sum to 1.

**Temperature sampling.** Generation works by computing `log P(w | ctx)` for every vocabulary word, dividing by temperature, shifting by the max for numerical stability, exponentiating, and normalising. Temperature < 1 sharpens the distribution (more predictable text); temperature > 1 flattens it (more random). This is the same mechanism used in neural language model decoding.

**OOV handling.** The tokenizer does not add unseen words to the vocabulary at inference time. With Laplace smoothing, an OOV word still receives probability `1 / (C(ctx) + V)` — small but finite, so perplexity stays finite. With KN, the continuation count for an unseen word is 0; the code inserts a tiny floor (`1e-10`) rather than hard zero so perplexity remains computable but becomes astronomically high.

---

## What's deliberately omitted

**BPE / subword tokenization.** This model uses whitespace tokenization. Real production n-gram models often operated on characters or morphemes; modern systems use BPE. Adding subword tokenization would obscure the n-gram logic without teaching anything new about smoothing.

**Backoff vs. interpolation.** Katz backoff (1987) uses the higher-order model when the count exceeds a threshold and *backs off* to the lower-order model otherwise. This implementation uses *interpolation* exclusively: all three levels always contribute. Interpolated KN consistently outperforms backoff KN in practice, but backoff is simpler to reason about.

**Good-Turing and absolute discounting.** Both are precursors to KN and are omitted to keep the code focused.

**Neural models.** The entire point of this implementation is to understand what neural models replaced and why. Backpropagation comes next.
