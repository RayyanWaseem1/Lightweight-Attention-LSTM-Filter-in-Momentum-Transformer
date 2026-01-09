#LSTM Models
#Contains vanilla LSTM and LSTM with lightweight attention

import torch
import torch.nn as nn
from typing import Optional, Tuple

class LSTMMomentum(nn.Module):
    #This is the standard vanilla LSTM for Deep Momentum Networks
    #Handles local processing of a short window, like a filter
    #Used as an encoder before being fed into the Momentum Transformer 

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 short_window: int = 63):
        
        super().__init__()
        self.short_window = short_window 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        #Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        #For standalone predictions (when not actively being used as an encoder)
        self.fc = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh() 

    def forward(self,
                x: torch.Tensor,
                return_sequences: bool = False) -> torch.Tensor:
        
        #x: [batch_size, seq_len, input_dim]
        #return_sequences: If True, return all LSTM outputs for the Transformer. If False, return final position predictions

        #If return_sequences = True [batch, short_window, hidden_dim]
        #If return_sequences = False [batch] positions in [-1,1]

        #Extracting the local window for short range processing
        x_local = x[:, -self.short_window:, :]

        #LSTM forward pass
        out, (h_n, c_n) = self.lstm(x_local)
        #out: [batch, short_window, hidden_dim]
        #h_n: [num_layers, batch, hidden_dim]

        if return_sequences:
            #return full sequence for Transformer integration
            return self.layer_norm(out)
        
        #using final hidden state for standalone prediction
        lstm_out = h_n[-1]
        lstm_out = self.layer_norm(lstm_out)

        #Project to position
        position = self.fc(lstm_out)
        position = self.tanh(position).squeeze(-1)

        return position 
    
class LSTMMMomentumDualPath(nn.Module):
    #LSTM with dual-path architecture. 
    #Main path is the standard LSTM encoding
    #Attention path focuses on particular key moments in short window 
    #Uses gated combination to preserve information while functioning as a filter

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 short_window: int = 63,
                 use_attention_path: bool = True):
        
        super().__init__()
        self.short_window = short_window
        self.hidden_dim = hidden_dim
        self.use_attention_path = use_attention_path 

        #Main LSTM encoder
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        if use_attention_path:
            #Using lightweight attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, 1)
            )

            #Gating mechanism to control the influence of attention
            #Model learns how much to rely on attention vs main path
            self.attention_gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

            self.layer_norm2 = nn.LayerNorm(hidden_dim)

        #For the standalone predictions
        self.fc = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self,
                x: torch.Tensor,
                return_sequences: bool = False,
                return_attention_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        #x: [batch, seq_len, input_dim]
        #return_sequences: If True, return all LSTM outputs for Transformer. If False, return final position predictions
        #return_attention_weights: If True, return attention weights for analysis

        #enhanced_sequence: [batch, short_window, hidden_dim] if return_sequences = True or [batch] positions if not
        #attention_weights: [batch, short_window, 1] or None

        #Extracting local window
        x_local = x[:, -self.short_window:, :] 

        #LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x_local)
        lstm_out = self.layer_norm1(lstm_out)

        attention_weights = None 

        if not self.use_attention_path:
            #Simple path: just return LSTM output
            if return_sequences:
                return lstm_out, None
            #Standalone predictions
            final_hidden = h_n[-1]
            position = self.fc(final_hidden)
            position = self.tanh(position).squeeze(-1)
            return position, None 
        
        #Enhanced path with the attention filtering
        attention_scores = self.attention(lstm_out)
        attention_weights_normalized = torch.softmax(attention_scores, dim=1)

        #Computing the attention context
        attention_context = torch.sum(
            attention_weights_normalized * lstm_out,
            dim = 1,
            keepdim = True
        )

        #Broadcasting the attention context to all of the timestamps
        attention_context = attention_context.expand(-1, self.short_window, -1)

        #Gating combination: model learns when to use the attention
        gate = self.attention_gate(lstm_out)

        #Combine: (1-gate) * original + gate * attention_context
        enhanced_out = (1-gate) * lstm_out + gate * attention_context
        enhanced_out = self.layer_norm2(enhanced_out)

        if return_sequences:
            #Return enhanced sequence for Transformer
            attn_weights = attention_weights_normalized.squeeze(-1) if return_attention_weights else None
            return enhanced_out, attn_weights
        
        #Standalone prediction using final timestamp
        final_repr = enhanced_out[:, -1, :]
        position = self.fc(final_repr)
        position = self.tanh(position).squeeze(-1)

        attn_weights = attention_weights_normalized.squeeze(-1) if return_attention_weights else None
        return position, attn_weights
    
class LSTMMomentumWithAttention(nn.Module):
    #Standalone LSTM with attention
    #This is used without feeding into the Transformer
    #Uses attention-weighted pooling over the entire sequence

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 short_window: int = 63):
        super().__init__()
        self.short_window = short_window
        self.hidden_dim = hidden_dim 

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0
        )

        #Attenntion over the entire sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        #Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #x: [batch, seq_len, input_dim]
        #return_attention: If True, also return the attention weights

        #Positions: [batch] in range [-1,1]
        #Attention weights: [batch, short_window] or None

        #Extract local window
        x_local = x[:, -self.short_window:, :]

        #LSTM forward pass
        lstm_out, _ = self.lstm(x_local)
        lstm_out = self.layer_norm1(lstm_out)

        #Compute the attention weights over the sequence
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim = 1)

        #Apply attention: weighted sum of the LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim = 1)
        context = self.layer_norm2(context)

        #Final position prediction
        position = self.fc(context)
        position = self.tanh(position).squeeze(-1)

        if return_attention:
            return position, attention_weights.squeeze(-1)
        return position, None 
    
def get_lstm_encoder(config, use_attention: bool = False):
    #Utility function to instantiate the appropriate LSTM encoder based on config 

    #Config: Model configuration 
    #Use attention: Whether to use the attention-enhanced LSTM 

    #Returns the LSTM encoder model 

    if use_attention:
        return LSTMMMomentumDualPath(
            input_dim = config.input_dim,
            hidden_dim = config.hidden_dim,
            num_layers = config.lstm_num_layers,
            dropout = config.lstm_dropout,
            short_window= config.short_window,
            use_attention_path=True
        )
    else:
        return LSTMMomentum(
            input_dim = config.input_dim,
            hidden_dim = config.hidden_dim,
            num_layers = config.lstm_num_layers,
            dropout = config.lstm_dropout,
            short_window= config.short_window
        )
    