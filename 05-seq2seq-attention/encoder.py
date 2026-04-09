import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    LSTM cell with explicit gate equations.

    Gate equations:
        f_t = sigmoid(Wf @ [h, x] + bf)
        i_t = sigmoid(Wi @ [h, x] + bi)
        g_t = tanh(Wg @ [h, x] + bg)
        o_t = sigmoid(Wo @ [h, x] + bo)
        c_t = f_t * c + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Each gate takes [h, x] as input, so input dim is hidden_size + input_size
        combined = hidden_size + input_size
        self.Wf = nn.Linear(combined, hidden_size)
        self.Wi = nn.Linear(combined, hidden_size)
        self.Wg = nn.Linear(combined, hidden_size)
        self.Wo = nn.Linear(combined, hidden_size)

    def forward(self, x, h, c):
        """
        x: (input_size,)
        h: (hidden_size,)
        c: (hidden_size,)
        Returns: h_t (hidden_size,), c_t (hidden_size,)
        """
        combined = torch.cat([h, x], dim=-1)   # (hidden_size + input_size,)

        f_t = torch.sigmoid(self.Wf(combined))
        i_t = torch.sigmoid(self.Wi(combined))
        g_t = torch.tanh(self.Wg(combined))
        o_t = torch.sigmoid(self.Wo(combined))

        c_t = f_t * c + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class BidirectionalEncoder(nn.Module):
    """
    Runs one LSTMCell left-to-right, another right-to-left, concatenates hidden states.

    Returns:
        all_hidden: (seq_len, 2*hidden_size)  — concatenated fwd+bwd at each position
        final_h:    (2*hidden_size,)           — concatenated final hidden states
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fwd_cell = LSTMCell(input_size, hidden_size)
        self.bwd_cell = LSTMCell(input_size, hidden_size)

    def forward(self, inputs):
        """
        inputs: (seq_len, input_size)
        """
        seq_len, input_size = inputs.shape
        device = inputs.device

        h_fwd = torch.zeros(self.hidden_size, device=device)
        c_fwd = torch.zeros(self.hidden_size, device=device)
        h_bwd = torch.zeros(self.hidden_size, device=device)
        c_bwd = torch.zeros(self.hidden_size, device=device)

        fwd_hiddens = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.fwd_cell(inputs[t], h_fwd, c_fwd)
            fwd_hiddens.append(h_fwd)

        bwd_hiddens = []
        for t in reversed(range(seq_len)):
            h_bwd, c_bwd = self.bwd_cell(inputs[t], h_bwd, c_bwd)
            bwd_hiddens.append(h_bwd)
        bwd_hiddens = list(reversed(bwd_hiddens))

        # Stack and concatenate forward + backward at each timestep
        fwd_stack = torch.stack(fwd_hiddens, dim=0)   # (seq_len, hidden_size)
        bwd_stack = torch.stack(bwd_hiddens, dim=0)   # (seq_len, hidden_size)
        all_hidden = torch.cat([fwd_stack, bwd_stack], dim=-1)  # (seq_len, 2*hidden_size)

        final_h = torch.cat([h_fwd, h_bwd], dim=-1)  # (2*hidden_size,)

        return all_hidden, final_h
