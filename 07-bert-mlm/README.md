# 07 — BERT MLM

## Context

Devlin et al. 2018 ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding") made a simple but profound bet: a model trained to fill in randomly masked words would learn representations useful for nearly every downstream NLP task.

The key insight is bidirectionality. Earlier language models (GPT-1, ELMo) could only condition on past tokens or concatenated two independently trained directional models. BERT attends in both directions simultaneously because it is never asked to *predict the next token* — only to reconstruct masked tokens given their full context. This lets position 5 attend to positions 6, 7, 8 … without leaking the answer, because position 5's token is masked away.

The result was a single pre-training recipe — train once on a large unlabelled corpus, then fine-tune on any downstream task with a small task-specific head — that set state of the art across eleven NLP benchmarks at the time of publication.

## What this code does

- `bert.py` — implements `TokenEmbedding`, `TransformerBlock` (bidirectional, PRE-LN), `BertEncoder`, `MLMHead` (with weight tying), and `BertMLM`. Also provides `mask_tokens`, which applies the 80/10/10 masking strategy and returns labels with -100 for unmasked positions.
- `train.py` — tokenises a short word-level corpus, trains a small `BertMLM` (d_model=64, 2 layers) for 1000 steps, and probes it: given "the [MASK] sat on the mat", prints the top-5 predictions for the masked position.
- `test_bert.py` — five pytest tests covering mask ratio, strategy distribution, weight tying, output shape, and loss behaviour.

## Key implementation details

**Weight tying.** `MLMHead.projection.weight` is set to the *same* `nn.Parameter` object as `TokenEmbedding.embedding.weight`. The input embedding maps token → vector and the output projection maps vector → token logits; they operate in the same semantic space. Tying them (a) halves the vocabulary-related parameter count and (b) regularises training: the embedding direction that represents "cat" on the input side also scores highly for "cat" on the output side. An `assert … is …` in `BertMLM.__init__` verifies this at construction time.

**The -100 label trick.** PyTorch's `CrossEntropyLoss` ignores positions where `label == -100`. In `mask_tokens`, unselected positions get label -100, so the loss is computed only on the 15% of tokens chosen for prediction. Without this, the gradient would be dominated by the trivial task of copying unmasked tokens.

**PRE-LN vs POST-LN.** The original 2017 transformer (Vaswani et al.) applied LayerNorm *after* the residual addition (POST-LN). This code uses PRE-LN: `x = x + sublayer(norm(x))`. PRE-LN moves LayerNorm inside the residual branch, which keeps the residual stream on a stable scale and makes gradient flow more predictable during deep-network training. It has become the near-universal standard (GPT-2, PaLM, LLaMA, etc.).

**80/10/10 masking.** Of the 15% selected positions: 80% become `[MASK]`, 10% become a random vocabulary token, and 10% are left unchanged. The 10%+10% non-mask replacements force the model to maintain useful representations at *all* positions, not just the ones visibly marked with `[MASK]` — at fine-tune time, no `[MASK]` tokens appear.

## What's deliberately omitted

**Next Sentence Prediction (NSP).** The original BERT paper included NSP as a second pre-training objective (predict whether sentence B follows sentence A). RoBERTa (Liu et al. 2019) showed NSP provides no consistent benefit and can hurt performance; it has been dropped in essentially all subsequent work.

**WordPiece tokenizer.** Real BERT uses subword tokenization to handle out-of-vocabulary words gracefully. This implementation uses word-level tokens to keep the focus on the model architecture.

**Positional and segment embeddings.** BERT adds three embeddings: token + position + segment (sentence A vs. B). This code adds only token + learned position embeddings. Segment embeddings are moot without NSP; sinusoidal vs. learned PE is a minor detail orthogonal to understanding MLM.

**Fine-tuning.** BERT's value comes from fine-tuning the pre-trained encoder on downstream tasks (classification, NER, QA) with a small task head. That step is not shown here — the focus is the pre-training mechanism.
