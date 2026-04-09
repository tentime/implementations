"""
N-gram Language Model implementation.

Covers unigram, bigram, and trigram models with Laplace and Kneser-Ney smoothing.
"""

import numpy as np
from collections import defaultdict


class Tokenizer:
    """
    Whitespace tokenizer that adds BOS/EOS boundary tokens and builds a vocabulary.

    Tokens are lowercased. The vocabulary maps token strings to integer IDs.
    OOV words are not added to the vocabulary at tokenize time; callers that need
    OOV handling should check membership before use.
    """

    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self):
        self.vocab = {}          # token -> id
        self.id_to_token = {}    # id -> token
        self._next_id = 0
        # Pre-register boundary tokens so they always have low IDs.
        self._add(self.BOS)
        self._add(self.EOS)

    def _add(self, token):
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1

    def tokenize(self, text, add_to_vocab=False):
        """
        Split *text* on whitespace, strip punctuation, lowercase.

        Returns a flat list of token strings with BOS prepended and EOS appended.
        If *add_to_vocab* is True, any unseen token is added to the vocabulary.
        """
        raw = text.split()
        tokens = [self.BOS]
        for word in raw:
            # Strip common punctuation from edges but keep internal apostrophes.
            cleaned = word.strip(".,!?;:\"()[]{}").lower()
            if not cleaned:
                continue
            if add_to_vocab:
                self._add(cleaned)
            tokens.append(cleaned)
        tokens.append(self.EOS)
        return tokens

    def vocab_size(self):
        return len(self.vocab)


class NgramLM:
    """
    Trigram language model with Laplace and Kneser-Ney smoothing.

    Internally stores:
      - unigram_counts[w]         : count of token w
      - bigram_counts[(w1,w2)]    : count of bigram (w1, w2)
      - trigram_counts[(w1,w2,w3)]: count of trigram
      - bigram_continuation[w2]   : number of unique left contexts for w2 (KN)
      - total_tokens              : total number of training tokens
    """

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        # Kneser-Ney: number of distinct bigrams that end with w (continuation count).
        self.bigram_continuation = defaultdict(set)  # word -> set of left contexts
        self.total_tokens = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus_text):
        """
        Tokenize *corpus_text* and accumulate n-gram counts.

        Sentences are separated by newlines; each line gets its own BOS/EOS.
        The vocabulary is built during this call.
        """
        for line in corpus_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            tokens = self.tokenizer.tokenize(line, add_to_vocab=True)
            self.total_tokens += len(tokens)

            for i, tok in enumerate(tokens):
                self.unigram_counts[tok] += 1
                if i >= 1:
                    self.bigram_counts[(tokens[i - 1], tok)] += 1
                    self.bigram_continuation[tok].add(tokens[i - 1])
                if i >= 2:
                    self.trigram_counts[(tokens[i - 2], tokens[i - 1], tok)] += 1

        # Convert continuation sets to counts now that training is done.
        self._kn_continuation = {
            w: len(ctxs) for w, ctxs in self.bigram_continuation.items()
        }
        self._total_bigram_types = len(self.bigram_counts)

    # ------------------------------------------------------------------
    # Laplace smoothing
    # ------------------------------------------------------------------

    def laplace_prob(self, context, word):
        """
        Laplace (add-one) smoothed trigram probability P(word | w1, w2).

        Formula:
            P_laplace(w3 | w1, w2) = (C(w1,w2,w3) + 1) / (C(w1,w2) + V)

        where V is the vocabulary size.  When the context bigram is unseen we
        fall back to the bigram Laplace estimate, and then to the unigram.

        Parameters
        ----------
        context : tuple of str
            The preceding two tokens (w1, w2).  Can also be a 1-tuple for bigram.
        word : str
            The word whose probability is being estimated.
        """
        V = self.tokenizer.vocab_size()
        if len(context) >= 2:
            w1, w2 = context[-2], context[-1]
            trigram_count = self.trigram_counts.get((w1, w2, word), 0)
            bigram_count = self.bigram_counts.get((w1, w2), 0)
            return (trigram_count + 1) / (bigram_count + V)
        elif len(context) == 1:
            w1 = context[-1]
            bigram_count = self.bigram_counts.get((w1, word), 0)
            unigram_count = self.unigram_counts.get(w1, 0)
            return (bigram_count + 1) / (unigram_count + V)
        else:
            unigram_count = self.unigram_counts.get(word, 0)
            return (unigram_count + 1) / (self.total_tokens + V)

    # ------------------------------------------------------------------
    # Kneser-Ney smoothing
    # ------------------------------------------------------------------

    def kneser_ney_prob(self, context, word, discount=0.75):
        """
        Interpolated Kneser-Ney smoothed trigram probability P(word | w1, w2).

        KN uses a *continuation* unigram probability instead of raw frequency:

            P_KN_unigram(w) = |{v : C(v, w) > 0}| / |{(u,v) : C(u,v) > 0}|

        This counts how many unique left contexts w appears in — a proxy for
        how likely w is to complete a new context.

        Bigram KN:
            P_KN(w | w1) = max(C(w1,w) - d, 0) / C(w1)
                           + lambda(w1) * P_KN_unigram(w)

            lambda(w1) = d * |{w : C(w1,w) > 0}| / C(w1)   (interpolation weight)

        Trigram KN:
            P_KN(w | w1,w2) = max(C(w1,w2,w) - d, 0) / C(w1,w2)
                              + lambda(w1,w2) * P_KN_bigram(w | w2)

        Parameters
        ----------
        context : tuple of str
        word : str
        discount : float
            Absolute discount parameter d (typically 0.75).
        """
        d = discount
        total_bigram_types = max(self._total_bigram_types, 1)

        # --- Unigram KN ---
        kn_unigram_count = self._kn_continuation.get(word, 0)
        p_kn_unigram = kn_unigram_count / total_bigram_types

        # Fallback: if word was never seen as a right-context, use a tiny floor.
        if p_kn_unigram == 0:
            p_kn_unigram = 1e-10

        if len(context) == 0:
            return p_kn_unigram

        # --- Bigram KN ---
        w1 = context[-1]
        bigram_count_w1w = self.bigram_counts.get((w1, word), 0)
        unigram_count_w1 = self.unigram_counts.get(w1, 0)

        if unigram_count_w1 == 0:
            p_kn_bigram = p_kn_unigram
        else:
            discounted = max(bigram_count_w1w - d, 0) / unigram_count_w1
            # Number of distinct words that follow w1
            n_w1_types = sum(
                1 for (a, b) in self.bigram_counts if a == w1
            )
            lambda_w1 = (d * n_w1_types) / unigram_count_w1
            p_kn_bigram = discounted + lambda_w1 * p_kn_unigram

        if len(context) < 2:
            return p_kn_bigram

        # --- Trigram KN ---
        w1, w2 = context[-2], context[-1]
        trigram_count = self.trigram_counts.get((w1, w2, word), 0)
        bigram_count_w1w2 = self.bigram_counts.get((w1, w2), 0)

        if bigram_count_w1w2 == 0:
            return p_kn_bigram

        discounted_tri = max(trigram_count - d, 0) / bigram_count_w1w2
        # Number of distinct words that follow (w1, w2)
        n_w1w2_types = sum(
            1 for (a, b, c) in self.trigram_counts if a == w1 and b == w2
        )
        lambda_w1w2 = (d * n_w1w2_types) / bigram_count_w1w2
        return discounted_tri + lambda_w1w2 * p_kn_bigram

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def perplexity(self, text, smoothing="laplace"):
        """
        Compute perplexity of *text* under this model.

        Perplexity = exp( -1/N * sum_i log P(w_i | w_{i-2}, w_{i-1}) )

        where N is the number of tokens (excluding BOS).  A lower perplexity
        means the model assigns higher probability to the text.

        Parameters
        ----------
        text : str
        smoothing : {'laplace', 'kneser_ney'}
        """
        tokens = self.tokenizer.tokenize(text, add_to_vocab=False)
        if len(tokens) < 2:
            return float("inf")

        prob_fn = self.laplace_prob if smoothing == "laplace" else self.kneser_ney_prob

        log_prob_sum = 0.0
        n = 0
        for i in range(1, len(tokens)):
            word = tokens[i]
            if i >= 2:
                ctx = (tokens[i - 2], tokens[i - 1])
            else:
                ctx = (tokens[i - 1],)

            p = prob_fn(ctx, word)
            if p <= 0:
                return float("inf")
            log_prob_sum += np.log(p)
            n += 1

        return float(np.exp(-log_prob_sum / n))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prefix_words, n_words=20, smoothing="laplace", temperature=1.0):
        """
        Generate text by sampling from the trigram distribution.

        The model builds a probability distribution over the full vocabulary for
        the current two-token context, then samples using temperature scaling:

            logits_i = log P(w_i | context)
            scaled_i = logits_i / temperature
            p_sample_i = softmax(scaled_i)

        Higher temperature -> flatter distribution (more random).
        Lower temperature -> sharper distribution (more greedy).

        Parameters
        ----------
        prefix_words : str
            Space-separated seed words (e.g. "to be").
        n_words : int
            Number of new words to generate (not counting the prefix).
        smoothing : {'laplace', 'kneser_ney'}
        temperature : float
            Sampling temperature > 0.

        Returns
        -------
        str
            The prefix words followed by generated words, space-joined.
        """
        prob_fn = self.laplace_prob if smoothing == "laplace" else self.kneser_ney_prob
        vocab_tokens = list(self.tokenizer.vocab.keys())
        # Remove boundary tokens from generation candidates.
        vocab_tokens = [t for t in vocab_tokens if t not in (Tokenizer.BOS, Tokenizer.EOS)]

        # Seed the context with BOS + prefix tokens.
        seed = self.tokenizer.tokenize(prefix_words, add_to_vocab=False)
        # Remove BOS/EOS added by tokenize so we can prepend BOS manually.
        seed = [t for t in seed if t != Tokenizer.EOS]
        if seed[0] != Tokenizer.BOS:
            seed = [Tokenizer.BOS] + seed

        generated = seed[1:]  # Strip the BOS for output, but keep it in context.
        context = seed

        for _ in range(n_words):
            if len(context) >= 2:
                ctx = (context[-2], context[-1])
            else:
                ctx = (context[-1],)

            # Build probability vector over vocab.
            log_probs = np.array([
                np.log(max(prob_fn(ctx, w), 1e-300))
                for w in vocab_tokens
            ])

            # Temperature scaling via log-space softmax.
            scaled = log_probs / temperature
            scaled -= scaled.max()  # Numerical stability.
            probs = np.exp(scaled)
            probs /= probs.sum()

            chosen = np.random.choice(vocab_tokens, p=probs)
            if chosen == Tokenizer.EOS:
                break
            generated.append(chosen)
            context.append(chosen)

        return " ".join(generated)
