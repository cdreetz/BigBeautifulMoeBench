# src/modules/router.py
import torch, torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, k, dtype=torch.float16):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False, dtype=dtype)
        self.k = k

    def forward(self, x):
        scores = self.proj(x).to(torch.float32)
        vals, idx = torch.topk(scores, self.k, dim=-1)
        w = torch.softmax(vals, dim=-1)
        return idx.to(torch.int32), w.to(torch.float32)

