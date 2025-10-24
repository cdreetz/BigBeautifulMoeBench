# src/modules/expert_mlp.py
import torch, torch.nn as nn
import torch.nn.functional as F

class ExpertMLP(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, dtype=torch.float16):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, d_ff, d_model, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, d_ff, dtype=dtype))
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)
        self.dtype = dtype

    @torch.inference_mode()
    def forward_grouped(self, x, topk_e, topk_w):
        S, H = x.shape
        K = topk_e.shape[1]
        y = torch.zeros((S, H), dtype=x.dtype, device=x.device)
        for k in range(K):
            e_idx_k = topk_e[:, k]
            w_k = topk_w[:, k].to(x.dtype)
            uniq_e = torch.unique(e_idx_k)
            for e in uniq_e.tolist():
                mask = (e_idx_k == e)
                if mask.sum() == 0: continue
                x_e = x[mask]
                W1 = self.w1[e]
                W2 = self.w2[e]
                h = F.silu(x_e @ W1.t())
                y_e = h @ W2.t()
                y_e *= w_k[mask].unsqueeze(1)
                y[mask] += y_e
        return y

