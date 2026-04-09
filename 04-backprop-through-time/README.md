# 04 — Backpropagation Through Time: RNN, LSTM, GRU

## Context

**Elman (1990) — Simple Recurrent Networks.** Jeff Elman's "Finding Structure in Time" introduced the SRN (now called vanilla RNN): a feedforward network with a copy of the hidden layer fed back as an additional input at the next timestep. This gave networks a form of memory. The problem emerged quickly in practice: training them on long sequences was unreliable. The reason is BPTT — gradients are multiplied by the weight matrix at every timestep. If the largest singular value of `W_hh` is < 1, gradient magnitudes shrink exponentially as they flow backward. If > 1, they explode. This is the vanishing/exploding gradient problem.

**Hochreiter & Schmidhuber (1997) — LSTM.** Sepp Hochreiter had diagnosed the vanishing gradient problem in his 1991 diploma thesis. The solution he and Schmidhuber published was the Long Short-Term Memory: a cell state `c_t` that flows through time with only additive updates, controlled by learned gates. Crucially, the gradient path through the cell state involves only element-wise multiplication by the forget gate `f_t` — no matrix multiply, no tanh saturation on the direct path. This creates what is sometimes called the "constant error carousel": gradients can flow backward through many timesteps without vanishing, as long as the forget gate stays near 1.

**Cho et al. (2014) — GRU.** The Gated Recurrent Unit is a simplified LSTM proposed by Kyunghyun Cho et al. in the same paper that introduced sequence-to-sequence learning. It merges the input and forget gates into a single update gate `z_t`, and eliminates the separate cell state — the hidden state itself carries long-range information. GRUs have fewer parameters than LSTMs and train faster; on most tasks they perform comparably. The choice between them is empirical.

**The vanishing gradient problem, simply.** Suppose you unroll 25 steps and the spectral norm of `W_hh` is 0.9. The gradient at step 1 has been multiplied by a factor that scales roughly as `0.9^24 ≈ 0.08`. At 50 steps it's `0.9^49 ≈ 0.005`. The network cannot learn dependencies longer than about 10 steps this way. LSTM avoids this because the dominant gradient path through `c_t` multiplies by `f_t` (a scalar per dimension), not by a full matrix.

---

## What this code does

Run `python train.py` and you will see:

1. **LSTM trains for 500 steps** on the Shakespeare snippet as a char-level language model. Every 100 steps: current loss and a 50-character greedy sample from the model. Early samples are noise; later samples begin to reproduce punctuation and common character patterns.
2. **Gradient norm comparison**: the script runs one BPTT pass through both a fresh VanillaRNN and a fresh LSTM on the same 25-char sequence, then prints the gradient norm at timestep t=1 and t=25 for each. The RNN ratio is typically orders of magnitude larger than the LSTM ratio — the gradient that was large at t=25 has shrunk dramatically by t=1 in the RNN, while the LSTM keeps them much more comparable.

---

## Key implementation details

**The cell state as a conveyor belt.** The LSTM cell state `c_t` is updated additively: `c_t = f_t * c_prev + i_t * g_t`. The gradient of the loss w.r.t. `c_{t-1}` passes through this equation as multiplication by `f_t` — a vector of values between 0 and 1, applied element-wise. There is no matrix multiply and no saturating nonlinearity on this direct path. This is why gradients can survive for dozens of timesteps.

**Why separate gate matrices.** One implementation choice (used here) is to have four separate weight matrices `W_f, W_i, W_g, W_o`, each of shape `(H, I+H)`. The alternative is a single concatenated matrix of shape `(4H, I+H)`. They are mathematically identical. Separate matrices are clearer for pedagogy because you can see which weights belong to which gate, and the backward pass is explicit about which gradient flows where. A single matrix is faster on hardware due to a single GEMM call.

**What the gradient norm comparison shows.** The numbers printed are the L2 norm of the gradient `dh` at different timesteps. In a fresh (untrained) VanillaRNN, the ratio of early-to-late gradient norms is already large — purely because of the random `W_hh`. In the LSTM, the forget gate (initialized near 1.0 due to `b_f = 1`) preserves the cell-state gradient, making early-timestep gradients much closer in magnitude to late-timestep ones. After training, LSTMs maintain this property when the forget gate learns to stay open for relevant information.

---

## What's deliberately omitted

- **Bidirectional RNNs.** A bidirectional RNN runs one forward and one backward pass over the sequence and concatenates the hidden states. This gives each position access to both past and future context — essential for tasks like named entity recognition. It is covered in folder 05 (attention and seq2seq).
- **Batching.** All implementations here process one sequence at a time. In practice, sequences are padded to the same length and processed in batches using packed/masked operations. NumPy can do this but the indexing becomes messier; the educational value is in the per-step equations.
- **PyTorch's cuDNN-optimized LSTM.** PyTorch's `nn.LSTM` uses a custom cuDNN kernel that fuses all four gate computations into a single kernel call on GPU. The result is 5–10x faster than naive separate matrix multiplies. The underlying math is identical to this implementation.
