#Momentum Transformer
#Full implementation of LSTM encoder (local) + Transformer (global)

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .LSTM import LSTMMomentum, LSTMMMomentumDualPath, get_lstm_encoder
from .Transformer_layers import TransformerEncoder, InterpretableMultiheadAttention, get_transformer_encoder

class MomentumTransformer(nn.Module):
    #Full Momentum Transformer model combining LSTM and Transformer encoder
    #LSTM encoder captures local dependencies + Transformer captures long-range dependencies

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 short_window: int = 63,
                 num_transformer_layers: int = 2,
                 num_attention_heads: int = 4,
                 transformer_dropout: float = 0.2,
                 feedforward_dim: int = 256,
                 use_lstm_attention: bool = False,
                 use_positional_encoding: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_lstm_attention = use_lstm_attention

        #LSTM Encoder for local feature extraction
        if use_lstm_attention:
            self.LSTM = LSTMMMomentumDualPath(
                input_dim = input_dim,
                hidden_dim = hidden_dim,
                num_layers = lstm_num_layers,
                dropout = lstm_dropout,
                short_window = short_window,
                use_attention_path=True
            )
        else:
            self.LSTM = LSTMMomentum(
                input_dim = input_dim,
                hidden_dim = hidden_dim,
                num_layers = lstm_num_layers,
                dropout = lstm_dropout,
                short_window = short_window
            )

        #Transformer Encoder for global feature extraction
        self.Transformer_layers = TransformerEncoder(
            d_model = hidden_dim,
            num_layers = num_transformer_layers,
            num_heads = num_attention_heads,
            dim_feedforward = feedforward_dim,
            dropout = transformer_dropout,
            use_positional_encoding = use_positional_encoding
        )

        #The prediction head 
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        #x: [batch, seq_len, input_dim]
        #return_attention: whether to return attention weights from LSTM attention path

        #positions: [batch, seq_len, hidden_dim]
        #attention_info: dict with LSTM and Transformer attention weights if requested

        #LSTM Encoding (short-range filtering)
        if self.use_lstm_attention:
            lstm_out, lstm_attention = self.LSTM(
                x,
                return_sequences = True,
                return_attention_weights = return_attention
            )
        else:
            lstm_out = self.LSTM(x, return_sequences = True)
            lstm_attention = None
        #lstm_out: [batch, seq_len, hidden_dim]

        #Transformer Encoding (long-range dependencies)
        transformer_out = self.Transformer_layers(lstm_out)
        #transformer_out: [batch, seq_len, hidden_dim]

        #Final timestep for prediction
        final_repr = transformer_out[:, -1, :] # [batch, hidden_dim]

        #Generate position
        position = self.prediction_head(final_repr).squeeze(-1) # [batch]

        if return_attention:
            attention_info = {
                'lstm_attention': lstm_attention,
            }
            return position, attention_info
        return position, None
    
class MomentumTransformerSimple(nn.Module):
    #Simplified vanilla LSTM + Transformer

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        self.model = MomentumTransformer(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            lstm_num_layers = 2,
            lstm_dropout = dropout,
            short_window= 63,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_heads,
            transformer_dropout=dropout,
            feedforward_dim=hidden_dim*4,
            use_lstm_attention=False,
            use_positional_encoding=True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [batch,seq_len,input_dim]

        #positions: [batch]

        position, _ = self.model(x, return_attention=False)
        return position
    

class MomentumTransformerDualPath(nn.Module):
    #Enhanced dual-path LSTM + Transformer model

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 use_lstm_attention: bool = True):
        super().__init__()

        self.model = MomentumTransformer(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            lstm_num_layers = 2,
            lstm_dropout = dropout,
            short_window = 63,
            num_transformer_layers = num_transformer_layers,
            num_attention_heads = num_heads,
            transformer_dropout = dropout,
            feedforward_dim = hidden_dim * 4,
            use_lstm_attention = use_lstm_attention, #Enhanced LSTM with attention
            use_positional_encoding = True
        )

    def forward(self, 
                x: torch.Tensor,
                return_attention: bool = False):
        
        #x: [batch, seq_len, input_dim]
        #return_attention: Whether to return the attention weghts 

        #positions: [batch]
        #attention_info: Dict - optional

        return self.model(x, return_attention = return_attention)
    
def get_momentum_transformer(config, model_type: str = 'enhanced'):
    #Factory function to create the Momentum Transformer based on configuration

    #Config: Model configuration
    #Model_type: 'simple' with no LSTM attention or 'enhanced' with LSTM attention

    #Returns the MomentumTransformer instance

    if model_type == 'simple':
        return MomentumTransformerSimple(
            input_dim = config.input_dim,
            hidden_dim = config.hidden_dim,
            num_transformer_layers = config.num_transformer_layers,
            num_heads = config.num_attention_heads,
            dropout = config.transformer_dropout
        )
    elif model_type == 'enhanced':
        return MomentumTransformerDualPath(
            input_dim = config.input_dim,
            hidden_dim = config.hidden_dim,
            num_transformer_layers = config.num_transformer_layers,
            num_heads = config.num_attention_heads,
            dropout = config.transformer_dropout,
            use_lstm_attention=config.use_lstm_attention
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    


if __name__ == "__main__":
    from Models.config import get_default_config

    config = get_default_config()
    config.model.input_dim = 32

    #creating model
    model = get_momentum_transformer(config.model, model_type = 'enhanced')

    #testing forward pass
    batch_size = 16
    seq_len = 252
    x = torch.randn(batch_size, seq_len, config.model.input_dim)

    positions, attention_info = model(x, return_attention = True)

    print(f"Input shape: {x.shape}")
    print(f"Output positions shape: {positions.shape}")
    print(f"Positions range: [{positions.min():.3f}, {positions.max():.3f}]")

    if attention_info and attention_info["lstm_attention"] is not None:
        print(f"LSTM attentions shape: {attention_info["lstm_attention"].shape}")

    #count parameters
    num_params = sum(p.num() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
