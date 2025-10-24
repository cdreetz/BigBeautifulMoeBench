# src/models/qwen3_moe_skeleton.py
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MoEConfig:
    d_model: int = 4096
    d_ff: int = 14336        # e.g., ~3.5x
    n_experts: int = 128     # Qwen3 MoE uses 128 experts
    top_k: int = 8           # Qwen3 MoE uses top-8 routing
    capacity_factor: float = 1.25
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        # x: [S, H]
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, k, dtype=torch.float16):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False, dtype=dtype)
        self.k = k

    @torch.no_grad()
    def capacity(self, tokens: int, n_experts: int, capacity_factor: float, k: int):
        # typical formula: ceil(k * tokens * cf / E)
        return int((k * tokens * capacity_factor + n_experts - 1) // n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # scores: [S, E] in fp32 for stability
        scores = self.proj(x).to(torch.float32)
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)        # [S,K]
        topk_w = torch.softmax(topk_vals, dim=-1)                        # [S,K]
        return topk_idx.to(torch.int32), topk_w.to(torch.float32)

class ExpertMLP(nn.Module):
    """
    Naive per-expert MLP bank. Baseline combines expert outputs weighted by router.
    Shapes:
      W1: [E, D_ff, H]
      W2: [E, H, D_ff]
    """
    def __init__(self, d_model, d_ff, n_experts, dtype=torch.float16):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, d_ff, d_model, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff, dtype=dtype))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)
        self.dtype = dtype

    @torch.inference_mode()
    def forward_grouped(self, x: torch.Tensor, topk_e: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
        """
        x:      [S, H] (fp16/fp32 on CUDA)
        topk_e: [S, K] int32 (expert indices)
        topk_w: [S, K] float32 (router weights over K)
        returns y: [S, H]
        """
        S, H = x.shape
        K = topk_e.shape[1]
        E = self.w1.shape[0]
        y = torch.zeros((S, H), dtype=x.dtype, device=x.device)

        # naive but robust: loop over k then experts present at that k
        # (fast enough for a baseline; you’ll replace with grouped kernels later)
        for k in range(K):
            e_idx_k = topk_e[:, k]            # [S]
            w_k = topk_w[:, k].to(x.dtype)    # [S]
            # process only experts that appear this step
            uniq_e = torch.unique(e_idx_k)
            for e in uniq_e.tolist():
                mask = (e_idx_k == e)
                if mask.sum() == 0:
                    continue
                x_e = x[mask]                 # [N_e, H]
                # expert e params
                W1 = self.w1[e]               # [D_ff, H]
                W2 = self.w2[e]               # [H, D_ff]
                # MLP: SiLU (good enough baseline; Qwen variants may use SwiGLU/GeGLU)
                h = F.silu(x_e @ W1.t())      # [N_e, D_ff]
                y_e = h @ W2.t()              # [N_e, H]
                # apply router weight AFTER MLP (activation breaks linearity)
                y_e *= w_k[mask].unsqueeze(1)
                y[mask] += y_e
        return y

class Qwen3MoELayer(nn.Module):
    """
    Minimal MoE layer for baseline inference:
      y = x + MoE( RMSNorm(x) )
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.rms = RMSNorm(cfg.d_model, eps=1e-6, dtype=cfg.dtype)
        self.router = TopKRouter(cfg.d_model, cfg.n_experts, cfg.top_k, dtype=cfg.dtype)
        self.experts = ExpertMLP(cfg.d_model, cfg.d_ff, cfg.n_experts, dtype=cfg.dtype)

    @torch.inference_mode()
    def forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S,H]
        x_norm = self.rms(x)
        topk_e, topk_w = self.router(x_norm)  # [S,K], [S,K]
        y = self.experts.forward_grouped(x_norm, topk_e, topk_w)  # [S,H]
        return x + y  # simple residual add to mimic transformer block structure

