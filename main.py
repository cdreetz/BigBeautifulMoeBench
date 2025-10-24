import json
import torch
from transformers import AutoTokenizer
from src.baseline import Qwen3ForCausalLM, Qwen3MoEConfig

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

with open("src/config.json") as f:
    raw = json.load(f)

cfg = Qwen3MoEConfig(
    vocab_size = raw["vocab_size"],
    hidden_size = raw["hidden_size"],
    intermediate_size = raw["intermediate_size"],
    num_hidden_layers = raw["num_hidden_layers"],
    num_attention_heads = raw["num_attention_heads"],
    num_key_value_heads = raw["num_key_value_heads"],
    head_dim = raw["head_dim"],
    num_experts = raw["num_experts"],
    num_experts_per_tok = raw["num_experts_per_tok"],
    rms_norm_eps = raw["rms_norm_eps"],
    rope_theta = raw["rope_theta"],
    norm_topk_prob = raw["norm_topk_prob"],
    router_aux_loss_coef = raw["router_aux_loss_coef"],
)

model = Qwen3ForCausalLM(cfg)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "what is 2+2?"

def chat(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    for _ in range(50):
        outputs = model(input_ids)
        next_token = outputs['logits'][0, -1].argmax()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(inputids[0])


if __name__ == "__main__":
    out = chat(text)
    print(out)
