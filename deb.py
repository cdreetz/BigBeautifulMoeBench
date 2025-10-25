import torch
from transformers import AutoTokenizer, AutoConfig
from src.baseline import Qwen3ForCausalLM, Qwen3MoEConfig
import safetensors.torch
from huggingface_hub import snapshot_download
import os

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Load config
hf_config = AutoConfig.from_pretrained(model_name)

cfg = Qwen3MoEConfig(
    vocab_size=hf_config.vocab_size,
    hidden_size=hf_config.hidden_size,
    intermediate_size=hf_config.intermediate_size,
    num_hidden_layers=hf_config.num_hidden_layers,
    num_attention_heads=hf_config.num_attention_heads,
    num_key_value_heads=hf_config.num_key_value_heads,
    head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
    num_experts=hf_config.num_experts,
    num_experts_per_tok=hf_config.num_experts_per_tok,
    max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 262144),
    rope_theta=hf_config.rope_theta,
    rms_norm_eps=hf_config.rms_norm_eps,
    hidden_act=getattr(hf_config, 'hidden_act', 'silu'),
    attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
    moe_intermediate_size=hf_config.moe_intermediate_size,
    shared_expert_intermediate_size=getattr(hf_config, 'shared_expert_intermediate_size', hf_config.intermediate_size),
    norm_topk_prob=getattr(hf_config, 'norm_topk_prob', False),
    router_aux_loss_coef=hf_config.router_aux_loss_coef,
    use_shared_expert=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
print("Loading model...")
with torch.device('meta'):
    model = Qwen3ForCausalLM(cfg)

# Load weights
cache_dir = snapshot_download(repo_id=model_name)
state_dict = {}
for file in sorted(os.listdir(cache_dir)):
    if file.endswith('.safetensors'):
        state_dict.update(safetensors.torch.load_file(
            os.path.join(cache_dir, file), 
            device=str(device)
        ))

model.load_state_dict(state_dict, assign=True, strict=False)

# Check what dtype the weights are
sample_weight = next(model.parameters())
print(f"Model weights dtype: {sample_weight.dtype}")
print(f"Model weights device: {sample_weight.device}")

model.eval()
print("Model loaded.\n")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Simple test
messages = [{"role": "user", "content": "what is 2+2?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"Input shape: {input_ids.shape}")
print(f"Input dtype: {input_ids.dtype}")

# Single forward pass
with torch.no_grad():
    print("\nRunning forward pass...")
    outputs = model(input_ids)
    logits = outputs['logits']
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits device: {logits.device}")
    print(f"Logits min: {logits.min().item():.4f}")
    print(f"Logits max: {logits.max().item():.4f}")
    print(f"Logits mean: {logits.mean().item():.4f}")
    print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"Logits has Inf: {torch.isinf(logits).any().item()}")
    
    # Try softmax
    last_logits = logits[0, -1, :].float()
    print(f"\nLast token logits shape: {last_logits.shape}")
    print(f"Last token logits min: {last_logits.min().item():.4f}")
    print(f"Last token logits max: {last_logits.max().item():.4f}")
    
    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    print(f"Probs min: {probs.min().item():.10f}")
    print(f"Probs max: {probs.max().item():.10f}")
    print(f"Probs sum: {probs.sum().item():.10f}")
    print(f"Probs has NaN: {torch.isnan(probs).any().item()}")
    print(f"Probs has Inf: {torch.isinf(probs).any().item()}")
    print(f"Probs has negative: {(probs < 0).any().item()}")
    
    # Try to sample
    print("\nAttempting to sample...")
    next_token = torch.multinomial(probs, num_samples=1)
    print(f"Sampled token: {next_token.item()}")
    print(f"Token text: {tokenizer.decode([next_token.item()])}")
