""" Transformer components for the global attention stage 

* **One mask convention** ``nn.MultiheadAttention`` treats ``True`` as
    *disallowed*; the hand-rolled attention used ``mask == 0`` as disallowed, so
    the same tensor passed to both would have masked opposite positions.
    Everything here now uses the PyTorch convention: ``True`` == masked out.
* **A causal mask is actually applied.** Attention within the window used to 
    be bidirectional. With only the final timestep ead that was not lookahead, 
    but it deviated from the paper and would become real leakage the moment
    per-timestep targets are introduced.
* **``InterpretableMultiheadAttention`` is now interpretable.** It previously 
    claimed to "share the value weights across heads" while using a 
    ``Linear(d_model, d_model)`` value projects reshaped per head -- i.e.
    ordinary multi-head attention. The TFT construction (Lim et al. 1912.09363)
    uses a *single shared value head* and averages attention across heads; that is
    what makes the weights readable. It is also wired into the encoder now, and
    the encoder can return its attention weights
* **``PositionalEncoding`` handles odd ``d_model``** instead of raising on a 
    shape mismatch
"""

from __future__ import annotations 

import math 
from typing import List, Literal, Optional, Tuple 

import torch
import torch.nn as nn 

def build_causal_mask(seq_len: int, device = None) -> torch.Tensor:
    """ Boolean causal mask of shape [seq_len, seq_len].
    
    ``mask[i,j] == True`` means position ``i`` may **not** attend to position
    ``j`` (PyTorch's ``attn_mask`` convention). Strictly upper-triangular, so 
    each position sees itself and everything before it
    """

    return torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool, device = device), diagonal = 1
    )

class PositionalEncoding(nn.Module):
    """ Sinusoidal positional encoding, valid for odd and even ``d_model``"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # for odd d_model the cosine block is one column shorter than div_term
        n_cos = pe[0, :, 1::2].shape[1]
        pe[0, :, 1::2] = torch.cos(position * div_term[:n_cos])

        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])

class InterpretableMultiheadAttention(nn.Module):
    """ Multi-head attention with a single shared value head (TFT-style).
    
    Queries and keys are projected per head, but **one** value projection of
    width ``d_k`` is shared across all heads and the per-head outputs are 
    averaged. Because every head writes into the same value space, the 
    head-averaged attention matrix is a meaningful "how much did position i use
    position j" map -- which is the whole point of the construction.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        self.scale = math.sqrt(self.d_k)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        # Shared across heads -- this is the interpretability mechanism 
        self.value_projection = nn.Linear(d_model, self.d_k)
        self.output_projection = nn.Linear(self.d_k, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        return_attention: bool = False,
    ):
        batch_size, seq_len, _ = x.shape

        q = self.query_projection(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1,2)
        k = self.key_projection(x).view(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(1,2)
        # [batch, 1, seq_len, d_k] -- broadcast to every head
        v = self.value_projection(x).unsqueeze(1)

        scores = torch.matmul(q, k.transpose(-2,-1)) / self.scale 

        if attention_mask is not None:
            mask = attention_mask 
            if mask.dtype != torch.bool:
                mask = mask != 0
            # True == disallowed, matching nn.MultiheadAttention
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim = -1)
        weights = self.dropout(weights)

        # [batch, heads, seq_len, d_k] -> average over heads 
        per_head = torch.matmul(weights, v)
        combined = per_head.mean(dim = 1)

        output = self.output_projection(combined)

        if return_attention:
            return output, weights.mean(dim = 1) # head-averaged, interpretable 
        return output, None 

class TransformerEncoderBlock(nn.Module):
    """ Pre-configurable encoder block: standard or interpretable attention"""

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dim_feedforward: int,
        dropout: float = 0.1,
        attention_type: Literal["standard", "interpretable"] = "interpretable",
    ):
        super().__init__()
        self.attention_type = attention_type 

        if attention_type == "interpretable":
            self.self_attn = InterpretableMultiheadAttention(d_model, num_heads, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim = d_model, num_heads = num_heads, dropout = dropout, batch_first = True
            )

        self.dropout = nn.Dropout(dropout)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.attention_type == "interpretable":
            attn_output, weights = self.self_attn(
                x, attention_mask = attention_mask, return_attention = return_attention
            )
        else:
            attn_output, weights = self.self_attn(
                x, x, x, attn_mask = attention_mask, need_weights = return_attention
            )

        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.feedforward(x))
        return x, weights 

class TransformerEncoder(nn.Module):
    """ Stack of encoder blocks with optional causal masking"""

    def __init__(
        self,
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        dim_feedforward: int, 
        dropout: float = 0.1, 
        use_positional_encoding: bool = True, 
        use_causal_mask: bool = True, 
        attention_type: Literal["standard", "interpretable"] = "interpretable",
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.use_causal_mask = use_causal_mask 

        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model = d_model,
                    num_heads = num_heads,
                    dim_feedforward = dim_feedforward,
                    dropout = dropout,
                    attention_type = attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        if attention_mask is None and self.use_causal_mask:
            attention_mask = build_causal_mask(x.size(1), x.device)

        attentions: Optional[List[torch.Tensor]] = [] if return_attention else None 
        for layer in self.layers:
            x, weights = layer(x, attention_mask, return_attention)
            if attentions is not None and weights is not None:
                attentions.append(weights)

        return self.norm(x), attentions 

def get_transformer_encoder(config) -> TransformerEncoder:
    """ Build a ``TransformerEncoder`` from a ``ModelConfig``"""
    return TransformerEncoder(
        d_model = config.hidden_dim,
        num_layers = config.num_transformer_layers,
        num_heads = config.num_attention_heads,
        dim_feedforward = config.feedforward_dim,
        dropout = config.transformer_dropout,
        use_positional_encoding = config.use_positional_encoding,
        use_causal_mask = config.use_causal_mask,
    )
