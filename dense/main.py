import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Qwen2Config
from models.model_1 import Qwen2ForCausalLM, generate_text


def benchmark_my_qwen(prompt, tokenizer, max_tokens=100):
    config = Qwen2Config()
    qwen = Qwen2ForCausalLM(config).cuda()
    qwen.load_state_dict(hf_model.state_dict())
    qwen.eval()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Warmup
    _ = generate_text(qwen, tokenizer, text, max_length=10)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    output = generate_text(qwen, tokenizer, text, max_length=max_tokens)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Count generated tokens
    input_tokens = len(tokenizer.encode(text))
    total_tokens = len(tokenizer.encode(output))
    generated_tokens = total_tokens - input_tokens
    
    elapsed_time = end_time - start_time
    tokens_per_second = generated_tokens / elapsed_time if elapsed_time > 0 else 0
    
    del qwen
    torch.cuda.empty_cache()
    
    return output, generated_tokens, elapsed_time, tokens_per_second



if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt= "Explain quantum computing in simple terms."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    my_times = []
    my_tok_per_sec = []
    print("Benchmarking custom implementation...")
    
    for i in range(3):
        output, gen_tokens, elapsed, tok_per_sec = benchmark_my_qwen(prompt, hf_model, tokenizer, max_tokens)
        my_times.append(elapsed)
        my_tok_per_sec.append(tok_per_sec)
        print(f"  Run {i+1}: {gen_tokens} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")
        if i == 0:  # Show output from first run
            print(f"  Output: {output[:100]}..." if len(output) > 100 else f"  Output: {output}")
    

