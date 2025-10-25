import os
import torch
import random
import numpy as np

def set_repro(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep perf
    torch.backends.cudnn.benchmark = True

def choose_dtype():
    # Prefer bfloat16 on modern GPUs, else float16
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # Hopper/Blackwell class -> bf16 is great
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32

def maybe_set_threads():
    # keep CPU helpers from oversubscribing
    cpu = os.cpu_count() or 8
    threads = max(1, cpu // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
