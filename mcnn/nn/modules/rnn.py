import torch
import torch.nn as nn
import complexPyTorch.complexFunctions as cf
from ..manifolds import create_manifold_parameter


class ManifoldRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, weight_manifold, layer=1, bias=True, nonlinearity=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.weight_manifold = weight_manifold
        self.layer = layer
        self.output_size = output_size

        self.wh0_transform = lambda x: x
        self.uh0_transform = lambda x: x
        self.wy0_transform = lambda x: x
        self.whi_transform = lambda x: x
        self.uhi_transform = lambda x: x
        self.wyi_transform = lambda x: x
        self.wq_transform = lambda x: x

        self.w_h = []
        self.u_h = []
        self.b_h = []

        wh0_flag, w_h0 = create_manifold_parameter(weight_manifold, (input_size, hidden_size))
        self.w_h.append(w_h0)
        uh0_flag, u_h0 = create_manifold_parameter(weight_manifold, (hidden_size, hidden_size))
        self.u_h.append(u_h0)

        if bias:
            b_h0 = nn.Parameter(torch.rand(hidden_size))
        else:
            b_h0 = torch.tensor(0, requires_grad=False)
        self.b_h.append(b_h0)

        if wh0_flag:
            self.wh0_transform = lambda x: x.transpose(-2, -1)
        if uh0_flag:
            self.uh0_transform = lambda x: x.transpose(-2, -1)

        for _ in range(layer-1):
            whi_flag, w_hi = create_manifold_parameter(weight_manifold, (hidden_size, hidden_size))
            self.w_h.append(w_hi)
            uhi_flag, u_hi = create_manifold_parameter(weight_manifold, (hidden_size, hidden_size))
            self.u_h.append(u_hi)

            if bias:
                b_hi = nn.Parameter(torch.rand(hidden_size))
            else:
                b_hi = torch.tensor(0, requires_grad=False)
            self.b_h.append(b_hi)

        if whi_flag:
            self.whi_transform = lambda x: x.transpose(-2, -1)
        if uhi_flag:
            self.uhi_transform = lambda x: x.transpose(-2, -1)

        if nonlinearity:
            self.func = nn.Tanh()
        else:
            self.func = cf.relu

        wq_flag, self.w_q = create_manifold_parameter(weight_manifold, (hidden_size, output_size))
        self.b_q = nn.Parameter(torch.zeros(output_size))

        if wq_flag:
            self.wq_transform = lambda x: x.transpose(-2, -1)

    def forward(self, x, h=None):
        seq_len = x.size(1)
        batch = x.size(0)

        if h == None:
            h = torch.zeros(batch, self.layer, self.hidden_size, dtype=x.dtype).to(x.device)
        out = torch.zeros(batch, seq_len, self.output_size, dtype=x.dtype).to(x.device)

        for i in range(seq_len):

            h_new = torch.zeros(batch, self.layer, self.hidden_size, dtype=x.dtype).to(x.device)
            h_new[:, 0] = self.func(torch.matmul(x[:, i, :], self.wh0_transform(self.w_h[0])) + torch.matmul(h[:, 0], self.uh0_transform(self.u_h[0])) + self.b_h[0])

            for j in range(1, self.layer):
                h_new[:, j] = self.func(torch.matmul(h_new[:, j-1], self.whi_transform(self.w_h[j])) + torch.matmul(h[:, j], self.uhi_transform(self.u_h[j])) + self.b_h[j])
            h = h_new

            out[:, i] = torch.matmul(h[:, -1], self.wq_transform(self.w_q)) + self.b_q

        return out, h
