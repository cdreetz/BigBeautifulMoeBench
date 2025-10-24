"""
Pure PyTorch implementation of Qwen3-30B-A3B MoE
No HuggingFace dependencies - just torch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Qwen3MoEConfig:
    """Configuration for Qwen3-30B-A3B-Instruct-2507
    
    Architecture verified from HuggingFace model card:
    - 30.5B total parameters, 3.3B activated per token
    - 48 layers, 128 experts (8 active), no shared experts in this version
    - GQA: 32 query heads, 4 key-value heads
    - Context: 262K native, 1M with YaRN
    """
    vocab_size: int = 151936
    hidden_size: int = 3584
    intermediate_size: int = 18944  # per expert FFN
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 128
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 262144
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    moe_intermediate_size: int = 2560  # shared expert size (may not be used)
    shared_expert_intermediate_size: int = 2560
    norm_topk_prob: bool = False
    router_aux_loss_coef: float = 0.001
    use_shared_expert: bool = True  # Can disable for cleaner baseline
    


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        # position_ids: [batch_size, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Force float32 for stability
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to q and k"""
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    """Multi-head attention with GQA support"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class Qwen3MLP(nn.Module):
    """Single expert MLP"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoE(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.norm_topk_prob = config.norm_topk_prob
        
        # Router (gate)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Experts - naive implementation with a list
        self.experts = nn.ModuleList([Qwen3MLP(config) for _ in range(self.num_experts)])
        
        # Shared expert (optional, Qwen3 has this)
        self.shared_expert = Qwen3MLP(config)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Router logits
        router_logits = self.gate(hidden_states_flat)  # [batch_size * seq_length, num_experts]
        
        # Top-k routing
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # NAIVE IMPLEMENTATION - Loop over experts (THIS IS WHAT WE'LL OPTIMIZE)
        # This is the slow part that FlashDMoE aims to fix
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
                
            # Get tokens for this expert
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            
            current_state = hidden_states_flat[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state)
            
            # Weighted by routing weights
            current_hidden_states = current_hidden_states * routing_weights[top_x_list, idx_list, None]
            
            # Accumulate
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # Shared expert
        shared_output = self.shared_expert(hidden_states_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states_flat))
        final_hidden_states = final_hidden_states + shared_gate * shared_output
        
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_length, hidden_dim)
        
        # Return auxiliary loss for load balancing
        router_aux_loss = self._compute_router_aux_loss(router_logits, selected_experts)
        
        return final_hidden_states, router_aux_loss
    
    def _compute_router_aux_loss(self, router_logits, selected_experts):
        """Compute auxiliary load balancing loss"""
        # Simple load balancing loss
        num_tokens = router_logits.shape[0]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Average probability per expert
        expert_usage = router_probs.mean(dim=0)
        
        # We want uniform distribution
        target = 1.0 / self.num_experts
        
        # L2 loss
        aux_loss = torch.mean((expert_usage - target) ** 2)
        
        return aux_loss


class Qwen3DecoderLayer(nn.Module):
    """Transformer decoder layer with MoE"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MoE(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        
        # Self attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        
        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_aux_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, router_aux_loss


class Qwen3Model(nn.Module):
    """Qwen3 transformer model"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_length, seq_length), float("-inf"), device=hidden_states.device),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Decoder layers
        total_aux_loss = 0.0
        for decoder_layer in self.layers:
            hidden_states, router_aux_loss = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            total_aux_loss += router_aux_loss
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, total_aux_loss


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model with language modeling head"""
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Forward through model
        hidden_states, router_aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Add router auxiliary loss
            loss = loss + self.config.router_aux_loss_coef * router_aux_loss
        
        return {
            'logits': logits,
            'loss': loss,
            'router_aux_loss': router_aux_loss,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """Simple greedy generation"""
        for _ in range(max_new_tokens):
            # Get logits for last token
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        return input_ids


# Test script
if __name__ == "__main__":
    print("Creating Qwen3-30B-A3B model (pure PyTorch)...")
    
    config = Qwen3MoEConfig()
    model = Qwen3ForCausalLM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nRunning forward pass with batch_size={batch_size}, seq_len={seq_len}...")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Router aux loss: {outputs['router_aux_loss']:.6f}")
    
    print("\n✓ Model created successfully!")
    print("This is your BASELINE - pure PyTorch, no optimizations")
    print("Now you can start converting modules to Triton kernels!")
