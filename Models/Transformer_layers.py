#Transformer Layers
#Contains the positional encoding and the transformer components for the long-range attention

import math
import torch
import torch.nn as nn
from typing import Optional 

class PositionalEncoding(nn.Module):
    #Sinusoidal positional encoding for transformer models

    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)

        #Creating the positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Precompute positional encodings
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        #Registering as buffer to avoid being considered as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [batch, seq_len, d_model]

        #x with positional encoding added [batch, seq_len, d_model]

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TransformerEncoderBlock(nn.Module):
    #A single transformer encoder block
    #Multi-headed attention + feedforward network with layer norm and residual connections

    def __init__(self,
                 d_model: int, 
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()

        #Multi-headed self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True
        )

        # Dropout for attention residual path
        self.dropout = nn.Dropout(dropout)

        #Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        #Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) ->torch.Tensor:
        
        #x: [batch, seq_len, d_model
        #attention_mask : [seq_len, seq_len] or None

        #output: [batch, seq_len, d_model]

        #Multi-headed self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(
            x, x, x, 
            attn_mask = attention_mask,
            need_weights = False
        )

        x = self.norm1(x + self.dropout(attn_output))

        #Feedforward with residual connection 
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)

        return x
    
class TransformerEncoder(nn.Module):
    #A stack of Transformer encoder blocks
    #Handling long-range dependencies via self-attention 

    def __init__(self,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)

        #Stack of transformer encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model = d_model,
                num_heads = num_heads,
                dim_feedforward = dim_feedforward,
                dropout = dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #x: [batch, seq_len, d_model]
        #attention_mask : [seq_len, seq_len] or None

        #output: [batch, seq_len, d_model]

        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        return x
    
class InterpretableMultiheadAttention(nn.Module):
    #Interpretable Multi-head attention
    #Shares the value weights across heads for interpretability
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 

        #seprate the query and key projections per head 
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)

        #Shared value projection across heads (for the interpretability)
        self.value_projection = nn.Linear(d_model, d_model)

        #Outputting projection
        self.output_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        #x: [batch, seq_len, d_model]
        #attention_mask : [seq_len, seq_len] or None

        #output: [batch, seq_len, d_model]

        batch_size, seq_len, _ = x.size()

        #Projecting queries, keys and values
        queries = self.query_projection(x)  # [batch, seq_len, d_model]
        keys = self.key_projection(x)       # [batch, seq_len, d_model]
        values = self.value_projection(x)   # [batch, seq_len, d_model]

        #Reshaping for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        values = values.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        #Queries, Keys, Values: [batch, num_heads, seq_len, d_k]

        #Calculating attention scores
        scores = torch.matmul(queries, keys.transpose(-2,-1)) / self.scale
        #Scores: [batch, num_heads, seq_len, seq_len]

        #Applying attention mask if provided
        if attention_mask is not None: 
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        #Calculating attention weights
        attention_weights = torch.softmax(scores, dim = -1)
        attention_weights = self.dropout(attention_weights)

        #Applying attention weights to values
        context = torch.matmul(attention_weights, values)
        #Context: [batch, num_heads, seq_len, d_k]

        #Reshape back
        context = context.transpose(1,2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        #Final output projection
        output = self.output_projection(context)

        if return_attention:
            return output, attention_weights
        return output
    
def get_transformer_encoder(config):
    #Utility function to create a TransformerEncoder from a config dictionary

    return TransformerEncoder(
        d_model = config.hidden_dim,
        num_layers = config.num_transformer_layers,
        num_heads = config.num_attention_heads,
        dim_feedforward=config.feedforward_dim,
        dropout = config.transformer_dropout,
        use_positional_encoding=config.use_positional_encoding
    )
