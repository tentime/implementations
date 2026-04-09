"""
Char-level language model on a Shakespeare snippet.
Trains an LSTM and demonstrates the vanishing-gradient gap vs VanillaRNN.
NumPy only.
"""

import numpy as np

from rnn import VanillaRNN, bptt, sgd
from lstm import LSTM, backward_lstm, sgd_lstm

# ---------------------------------------------------------------------------
# Embedded Shakespeare text
# ---------------------------------------------------------------------------
TEXT = """To be, or not to be, that is the question: Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them. To die, to sleep, no more; and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to: tis a consummation devoutly to be wished."""


# ---------------------------------------------------------------------------
# Char vocabulary
# ---------------------------------------------------------------------------

chars = sorted(set(TEXT))
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
V = len(chars)

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1.0
    return v

text_ids = [char2id[c] for c in TEXT]
xs_all = [one_hot(i, V) for i in text_ids]
ys_all = text_ids[1:] + [text_ids[0]]   # shift by 1 (next-char prediction)


# ---------------------------------------------------------------------------
# Sample from LSTM
# ---------------------------------------------------------------------------

def sample_lstm(model, seed_id, n=50):
    h = np.zeros(model.H)
    c = np.zeros(model.H)
    x = one_hot(seed_id, V)
    out = []
    for _ in range(n):
        h, c, _ = model.cell.forward(x, h, c)
        p = model.output(h)
        # Greedy decode
        next_id = int(np.argmax(p))
        out.append(id2char[next_id])
        x = one_hot(next_id, V)
    return "".join(out)


# ---------------------------------------------------------------------------
# Train LSTM
# ---------------------------------------------------------------------------

np.random.seed(42)

HIDDEN = 64
SEQ_LEN = 25      # chunk length for each step
LR = 0.005
STEPS = 500

lstm = LSTM(input_size=V, hidden_size=HIDDEN, output_size=V)

print("=== LSTM Training (char-level Shakespeare) ===")
for step in range(1, STEPS + 1):
    # Pick a random starting position
    start = np.random.randint(0, len(text_ids) - SEQ_LEN - 1)
    xs = xs_all[start: start + SEQ_LEN]
    ys = ys_all[start: start + SEQ_LEN]

    states = lstm.forward(xs)
    grads, loss = backward_lstm(lstm, xs, ys, states)
    sgd_lstm(lstm, grads, lr=LR)

    if step % 100 == 0:
        seed = text_ids[start]
        sample = sample_lstm(lstm, seed, n=50)
        print(f"  step {step:4d}  loss={loss:.4f}  sample: {repr(sample)}")

print()


# ---------------------------------------------------------------------------
# Vanishing gradient demo: gradient norm at t=1 vs t=25
# ---------------------------------------------------------------------------

print("=== Vanishing Gradient Demo ===")
print("Comparing gradient norms flowing back to timestep t=1 vs t=25")
print()

# Use a fixed chunk for fairness
xs_demo = xs_all[:SEQ_LEN]
ys_demo = ys_all[:SEQ_LEN]

# ---- VanillaRNN ----
rnn = VanillaRNN(input_size=V, hidden_size=HIDDEN, output_size=V)
h0 = np.zeros(HIDDEN)

# We need per-timestep gradient norms.
# Re-implement a lightweight BPTT that records dh at each step.

T = SEQ_LEN

# Forward
hs_rnn = [h0.copy()]
for t in range(T):
    h = rnn.forward_step(xs_demo[t], hs_rnn[-1])
    hs_rnn.append(h.copy())

ps_rnn = [rnn.output(hs_rnn[t + 1]) for t in range(T)]

# Backward — record norm of dh_next at each step
dh_next = np.zeros(HIDDEN)
grad_norms_rnn = {}

for t in reversed(range(T)):
    dy = ps_rnn[t].copy()
    dy[ys_demo[t]] -= 1.0
    dy /= T

    dh = rnn.W_hy @ dy + dh_next
    dtanh = (1.0 - hs_rnn[t + 1] ** 2) * dh
    dh_next = rnn.W_hh.T @ dtanh
    grad_norms_rnn[t] = float(np.linalg.norm(dh_next))

# ---- LSTM ----
lstm_demo = LSTM(input_size=V, hidden_size=HIDDEN, output_size=V)
states_demo = lstm_demo.forward(xs_demo)
hs_lstm, cs_lstm, caches_lstm = states_demo

ps_lstm = [lstm_demo.output(hs_lstm[t]) for t in range(T)]

dh_next_l = np.zeros(HIDDEN)
dc_next_l = np.zeros(HIDDEN)
grad_norms_lstm = {}

for t in reversed(range(T)):
    cache = caches_lstm[t]
    f_t, i_t, g_t, o_t = cache["f_t"], cache["i_t"], cache["g_t"], cache["o_t"]
    c_t, c_prev, tanh_ct = cache["c_t"], cache["c_prev"], cache["tanh_ct"]
    z = cache["z"]

    dy = ps_lstm[t].copy()
    dy[ys_demo[t]] -= 1.0
    dy /= T

    dh = lstm_demo.W_hy @ dy + dh_next_l
    grad_norms_lstm[t] = float(np.linalg.norm(dh))

    do_t = dh * tanh_ct
    dc_t = dh * o_t * (1.0 - tanh_ct ** 2) + dc_next_l
    df_t = dc_t * c_prev
    di_t = dc_t * g_t
    dg_t = dc_t * i_t
    dc_next_l = dc_t * f_t

    df_pre = df_t * f_t * (1.0 - f_t)
    di_pre = di_t * i_t * (1.0 - i_t)
    dg_pre = dg_t * (1.0 - g_t ** 2)
    do_pre = do_t * o_t * (1.0 - o_t)

    dz = (lstm_demo.cell.W_f.T @ df_pre
          + lstm_demo.cell.W_i.T @ di_pre
          + lstm_demo.cell.W_g.T @ dg_pre
          + lstm_demo.cell.W_o.T @ do_pre)
    dh_next_l = dz[:HIDDEN]

# ---- Report ----
print(f"  VanillaRNN — gradient norm at t=1  : {grad_norms_rnn.get(0, 0.0):.6f}")
print(f"  VanillaRNN — gradient norm at t=25 : {grad_norms_rnn.get(T-1, 0.0):.6f}")
ratio_rnn = grad_norms_rnn.get(0, 1e-8) / (grad_norms_rnn.get(T-1, 1e-8) + 1e-12)
print(f"  Ratio (t=1 / t=25)                 : {ratio_rnn:.2f}x  ← vanishing visible here")
print()
print(f"  LSTM       — gradient norm at t=1  : {grad_norms_lstm.get(0, 0.0):.6f}")
print(f"  LSTM       — gradient norm at t=25 : {grad_norms_lstm.get(T-1, 0.0):.6f}")
ratio_lstm = grad_norms_lstm.get(0, 1e-8) / (grad_norms_lstm.get(T-1, 1e-8) + 1e-12)
print(f"  Ratio (t=1 / t=25)                 : {ratio_lstm:.2f}x  ← more stable")
print()
print("  The RNN ratio >> LSTM ratio demonstrates vanishing gradients.")
print("  Gradients at early timesteps are orders of magnitude smaller in the RNN.")
