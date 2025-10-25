import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Callable

from config import Qwen2Config


@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_row = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask)

    x_squared = x_row * x_row
    var = tl.sum(x_squared, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = (x_row * rstd) * weight

    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)

def triton_rmsnorm(x, weight, eps=1e-06):
    batch_size, seq_len, hidden_size = x.shape
    n_rows = batch_size * seq_len

    x_flat = x.view(n_rows, hidden_size)
    output = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (n_rows, )

    rmsnorm_kernel[grid](
        x_flat, weight, output,
        n_rows, hidden_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view(batch_size, seq_len, hidden_size)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
