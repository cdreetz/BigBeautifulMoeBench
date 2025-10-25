import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEFAULT_MODEL = os.environ.get(
    "BBMB_MODEL_ID",
    # Small MoE by default (fits on a single strong GPU). Swap to Qwen official if you like.
    "huihui-ai/Huihui-MoE-1B-A0.6B"
)

def _quant_config():
    use_4bit = os.environ.get("BBMB_4BIT", "1") == "1"
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_quant_type="nf4",
    )

def load_baseline(model_id: str = DEFAULT_MODEL, dtype=None, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    quant = _quant_config()
    if quant is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=quant,
            trust_remote_code=True,
        )
    model.eval()
    return tok, model

@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 16):
    # Try to use the chat template if available (most Qwen3 variants provide it)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if model.device.type == "cuda":
        input_ids = input_ids.to(model.device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # deterministic
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
    )
    # Slice the new tokens only
    gen_ids = out[:, input_ids.shape[-1]:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return text
