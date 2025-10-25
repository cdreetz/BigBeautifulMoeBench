import argparse
from src.utils.init_env import set_repro, choose_dtype, maybe_set_threads
from src.models.qwen3_baseline import load_baseline, generate, DEFAULT_MODEL
from src.benchmark_utils import time_fn, report_latency_ms

def parse_args():
    p = argparse.ArgumentParser(description="BigBeautifulMoeBench baseline inference")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="HF model id (env BBMB_MODEL_ID overrides).")
    p.add_argument("--prompt", type=str, default="What is 2+2?",
                   help="User prompt to run through the model.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--iters", type=int, default=10, help="Benchmark iterations.")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quant.")
    return p.parse_args()

def main():
    args = parse_args()
    set_repro(1234)
    maybe_set_threads()
    if args.no_4bit:
        import os
        os.environ["BBMB_4BIT"] = "0"

    dtype = choose_dtype()
    print(f"[init] dtype={dtype}, model={args.model}")
    tok, model = load_baseline(args.model, dtype=dtype)

    # Define a callable for benchmarking
    def infer():
        return generate(model, tok, args.prompt, max_new_tokens=args.max_new_tokens)

    # One correctness pass
    out = infer()
    print("\n### Output")
    print(out)

    # Quick latency sample
    lat = time_fn(infer, warmup=args.warmup, iters=args.iters)
    report_latency_ms("baseline.generate", lat)

if __name__ == "__main__":
    main()
