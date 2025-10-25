import time
import torch

def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_fn(fn, *args, warmup=5, iters=20, **kwargs):
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    cuda_synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args, **kwargs)
    cuda_synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def report_latency_ms(name, latency_s):
    ms = latency_s * 1000
    print(f"[latency] {name}: {ms:.2f} ms/iter")
