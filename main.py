import torch
from transformers import AutoTokenzier
from src.baseline import Qwen3ForCausalLM, Qwen3MoEConfig


config = Qwen3MoEConfig(vocab_size=1000, hidden_size=512, num_hidden_layers=2, num_experts=4)
model = Qwen3ForCausalLM(config)
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
