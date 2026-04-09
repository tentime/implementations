import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Special token IDs (used throughout)
PAD_ID = 0
MASK_ID = 1
BOS_ID = 2
EOS_ID = 3


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_k)
        mask: optional boolean tensor broadcastable to (batch, heads, seq_q, seq_k)
              True  = keep this position
              False = mask it out (set to -inf before softmax)
    Returns:
        output:  (batch, heads, seq_q, d_k)
        weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


class TokenEmbedding(nn.Module):
    """Simple learned token embedding."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids):
        # Scale by sqrt(d_model) — standard practice from Vaswani et al.
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class TransformerBlock(nn.Module):
    """
    Bidirectional transformer block (no causal mask).
    Multi-head self-attention (scaled dot-product written out manually,
    not nn.MultiheadAttention) + Add & Norm + FFN + Add & Norm.

    Uses PRE-LayerNorm (modern standard): LayerNorm is applied BEFORE the
    sub-layer rather than after. This is the GPT-2/PaLM convention and
    differs from the original Vaswani et al. 2017 paper's POST-LN.
    PRE-LN trains more stably at depth because gradients flow through the
    residual branch without passing through the norm first.
    """

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Per-head projections — kept explicit (same style as folder 06)
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward network: two linear layers with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # PRE-LN: norms applied before each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _self_attention(self, x, mask=None):
        """Multi-head self-attention written out explicitly."""
        head_outputs = []
        for h in range(self.num_heads):
            Q = self.W_q[h](x).unsqueeze(1)  # (batch, 1, seq, d_k)
            K = self.W_k[h](x).unsqueeze(1)
            V = self.W_v[h](x).unsqueeze(1)
            out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
            head_outputs.append(out.squeeze(1))  # (batch, seq, d_k)
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq, d_model)
        return self.W_o(concat)

    def forward(self, x, mask=None):
        """
        x:    (batch, seq_len, d_model)
        mask: optional attention mask (bidirectional — None means attend everywhere)
        """
        # PRE-LN attention sub-layer: norm → attention → residual
        x = x + self._self_attention(self.norm1(x), mask=mask)
        # PRE-LN FFN sub-layer: norm → FFN → residual
        x = x + self.ffn(self.norm2(x))
        return x


class BertEncoder(nn.Module):
    """
    Stack of TransformerBlock layers. No causal mask — attends to all positions.
    This bidirectional attention is the defining property of BERT-style encoders.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        # Final norm after all layers (standard for PRE-LN stacks)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class MLMHead(nn.Module):
    """
    Prediction head for masked language modeling.
    Projects hidden states to vocabulary size.

    Weight tying: the projection weight matrix is the SAME nn.Parameter as the
    token embedding matrix — a reference is stored, not a copy. This means
    self.projection.weight IS self.token_embedding.embedding.weight.

    Why tie weights? The input embedding maps token → vector and the output
    projection maps vector → token scores. They operate in the same semantic
    space, so sharing parameters (a) reduces parameter count and (b) regularises
    training: the same direction in embedding space that means "cat" on the input
    side also scores highly for "cat" on the output side.
    """

    def __init__(self, d_model, vocab_size, token_embedding):
        super().__init__()
        # Small transform before the final projection (standard in BERT)
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Linear projection: hidden → vocab logits
        self.projection = nn.Linear(d_model, vocab_size, bias=True)
        # Weight tying: share the embedding matrix — same object, not a copy
        self.projection.weight = token_embedding.embedding.weight

    def forward(self, x):
        x = self.transform(x)
        return self.projection(x)  # (batch, seq_len, vocab_size)


class BertMLM(nn.Module):
    """
    Full BERT-style masked language model.

    Architecture:
        TokenEmbedding → BertEncoder (bidirectional) → MLMHead

    Positional embeddings are kept simple (learned, added to token embeddings)
    to focus on the core MLM mechanism. WordPiece and segment embeddings
    are omitted — see README for what's deliberately left out.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # Learned positional embeddings (one vector per position up to max_len)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.encoder = BertEncoder(num_layers, d_model, num_heads, d_ff)
        # MLMHead receives a reference to token_embedding for weight tying
        self.mlm_head = MLMHead(d_model, vocab_size, self.token_embedding)

        self._init_weights()

        # Verify weight tying at construction time
        assert self.mlm_head.projection.weight is self.token_embedding.embedding.weight, (
            "Weight tying failed: MLMHead.projection.weight is not the same object "
            "as TokenEmbedding.embedding.weight"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len)
        Returns logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        x = self.encoder(x)
        logits = self.mlm_head(x)
        return logits


def mask_tokens(token_ids, vocab_size, mask_prob=0.15, mask_id=MASK_ID, seed=None):
    """
    Apply the BERT masking strategy to a batch of token ids.

    Of all tokens, 15% are selected for prediction.
    Of those selected:
      - 80% are replaced with MASK_ID
      - 10% are replaced with a random vocabulary token
      - 10% are kept unchanged

    The -100 label trick: CrossEntropyLoss ignores positions with label=-100,
    so we only compute loss on the 15% of selected (masked) positions.
    This is crucial — without it the model would be penalised for correctly
    reproducing unmasked tokens, which is trivial and dominates the gradient.

    Args:
        token_ids:  (batch, seq_len) integer tensor
        vocab_size: size of the vocabulary (for random token sampling)
        mask_prob:  fraction of tokens to select (default 0.15)
        mask_id:    token id used as the [MASK] token
        seed:       optional int for reproducibility

    Returns:
        masked_ids: (batch, seq_len) — modified token ids
        labels:     (batch, seq_len) — -100 for unselected, true id for selected
    """
    if seed is not None:
        torch.manual_seed(seed)

    masked_ids = token_ids.clone()
    labels = torch.full_like(token_ids, -100)  # -100 = ignore in cross-entropy

    # Draw a uniform random value for each token position
    rand = torch.rand(token_ids.shape)

    # Select 15% of positions for prediction
    selected = rand < mask_prob

    # Record the true token ids for selected positions
    labels[selected] = token_ids[selected]

    # Among selected positions, decide replacement strategy
    # Draw another random value only for the selected subset
    strategy = torch.rand(token_ids.shape)

    # 80% → replace with MASK_ID
    mask_replace = selected & (strategy < 0.80)
    masked_ids[mask_replace] = mask_id

    # 10% → replace with a random token from the vocabulary
    random_replace = selected & (strategy >= 0.80) & (strategy < 0.90)
    num_random = random_replace.sum().item()
    if num_random > 0:
        random_tokens = torch.randint(0, vocab_size, (num_random,), dtype=token_ids.dtype)
        masked_ids[random_replace] = random_tokens

    # 10% → keep unchanged (no action needed; masked_ids is already a clone)

    return masked_ids, labels
