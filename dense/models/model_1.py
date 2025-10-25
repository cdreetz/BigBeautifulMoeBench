import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Callable

from config import Qwen2Config

from models.components.norms.triton_norm import RMSNorm
from models.components.embeddings.rotary_embedding import Qwen2RotaryEmbedding
from models.components.attentions.base_attention import Attention
from models.components.decoder import DecoderLayer



class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.current_pos = 0

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_cache()
        self.current_pos = 0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            if use_cache:
                pos_ids = torch.arange(
                    self.current_pos, self.current_pos + input_embeds.shape[1],
                    device=input_embeds.device, dtype=torch.long
                ).unsqueeze(0)
                self.current_pos += input_embeds.shape[1]
                position_ids = pos_ids
            else:
                position_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)

        if use_cache and hasattr(self.layers[0].self_attn, 'cache_k') and self.layers[0].self_attn.cache_k is not None:
            query_len = input_embeds.shape[1]
            key_len = self.layers[0].self_attn.cache_k.shape[2] + query_len
            causal_mask = create_causal_mask(key_len, input_embeds.device, input_embeds.dtype)
            # Only use the part of the mask relevant to the current query
            causal_mask = causal_mask[:, :, -query_len:, :]
        else:
            seq_len = input_embeds.shape[1]
            causal_mask = create_causal_mask(seq_len, input_embeds.device, input_embeds.dtype)

        causal_mask_mapping = {
            "full_attention": causal_mask,
            "sliding_attention": causal_mask,
        }

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        return logits

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    original_length = input_ids.shape[1]

    model.model.reset_kv_cache()
    generated_tokens = []

    with torch.no_grad():
        # First pass with full prompt
        logits = model(input_ids, use_cache=True)
        next_token_logits = logits[0, -1, :].clone()

        if generated_tokens:
            for token_id in set(generated_tokens):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        next_token_logits = next_token_logits / temperature

        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token_id.item())

        if next_token_id.item() == tokenizer.eos_token_id:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text

        # Continue generating token by token
        for _ in range(max_length - 1):
            logits = model(next_token_id.unsqueeze(0), use_cache=True)
            next_token_logits = logits[0, -1, :].clone()

            if generated_tokens:
                for token_id in set(generated_tokens):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty

            next_token_logits = next_token_logits / temperature

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token_id.item())

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
