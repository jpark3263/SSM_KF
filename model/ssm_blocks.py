"""Pure-PyTorch Mamba-1 block (no CUDA/Triton kernels).

Structure: in_proj -> causal depthwise conv1d -> SiLU -> selective scan -> SiLU(z) gate -> out_proj.
Parameter names match mamba_ssm.Mamba, so checkpoints trained with the CUDA kernels load directly.
The block operates on [B, T, d_model] and returns [B, T, d_model].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False, **fk)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True, **fk)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **fk)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **fk)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **fk)
        self._init_dt(self.dt_proj, self.d_inner, dt_min, dt_max, dt_init_floor)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

    @staticmethod
    def _init_dt(dt_proj, d_inner, dt_min, dt_max, dt_init_floor):
        dt_init_std = dt_proj.in_features ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt.to(dt_proj.bias.device))
        dt_proj.bias._no_reinit = True

    def _selective_scan(self, x):
        """x: [B, T, d_inner] (after conv + SiLU). Returns y: [B, T, d_inner]."""
        Bsz, T, D = x.shape
        x_dbl = self.x_proj(x)  # [B, T, dt_rank + 2N]
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, T, D]  (dt_proj bias included, then softplus)
        A = -torch.exp(self.A_log.float())  # [D, N]
        dA = torch.exp(dt.unsqueeze(-1) * A)  # [B, T, D, N]
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)  # [B, T, D, N]
        h = x.new_zeros(Bsz, D, self.d_state)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        y = torch.stack(ys, dim=1)
        return y + x * self.D

    def forward(self, hidden):
        """hidden: [B, T, d_model] -> [B, T, d_model]"""
        Bsz, T, _ = hidden.shape
        xz = self.in_proj(hidden)  # [B, T, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[..., :T].transpose(1, 2)
        x = self.act(x)
        y = self._selective_scan(x)
        y = y * F.silu(z)
        return self.out_proj(y)
