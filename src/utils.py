from transfomers import AutoTokenizer

def get_toks(
    text: str, 
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model(input)
