""" LSTM encoders: the *local* filter that feeds the Transformer 

Three modes are now available via ``ModelConfig.lstm_transformer_mode``:

``"strided"`` (default)
    Split the sequence into consecutive ``short_window`` blocks, encode each 
    block independently with the shared LSTM, and emit one embedding per block.
    The Transformer then attends across those local summaries. This is 
    literally "lightweight local filter + global attention", and it is the only 
    mode in which both halves of the architecture do the job they are named for.

``"full"``
    Run the LSTM across the entire sequence and let the Transformer attend over
    every timestep. Closes to Wood et al. (2112.08534); ~4x the LSTM compute

``"truncate"``
    The legacy behavior, kept so the regression can be measured rather than 
    asserted. Pair it with ``sequence_length == short_window`` so bars are not 
    loaded and discarded.

Ablation parity:
    The vanilla and attention encoders are the *same class* with 
    ``use_attention_path`` toggled, so they share an identical prediciton head. 
    Previously the vanilla head was ``Linear(hidden, 1)`` while the attention head
    was ``Linear(hidden, hidden//2) -> ReLU -> Dropout -> Linear(hidden//2, 1)``, 
    which confounded "attention helps" with "two extra layers help'>
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple 

import torch
import torch.nn as nn 

EncoderMode = Literal["strided", "full", "truncate"]

class LSTMMomentumEncoder(nn.Module):
    """ Shared LSTM encoder with an optional lightweight attention path."""

    def __init__(
        self,
        input_dim: int, 
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        short_window: int = 63,
        use_attention_path: bool = False,
        mode: EncoderMode = "strided",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.short_window = short_window 
        self.use_attention_path = use_attention_path 
        self.mode = mode 

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True, 
            dropout = dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        if use_attention_path:
            # Lightweight additive attention over the local window
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, 1),
            )
            # Gate lets the model choose how much to trust the attention path.
            self.attention_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Identical head in both variants -- see "Ablation parity" above
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.tanh = nn.Tanh()

    # Internals 
    def _encode_window(
        self, x: torch.Tensor, return_attention: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Encode one window: [N, W, F] -> ([N, W, H], attn or None)."""
        out, _ = self.lstm(x)
        out = self.layer_norm(out)

        if not self.use_attention_path:
            return out, None 

        scores = self.attention(out)
        weights = torch.softmax(scores, dim = 1)
        context = torch.sum(weights * out, dim = 1, keepdim = True).expand_as(out)

        gate = self.attention_gate(out)
        enhanced = self.layer_norm2((1 - gate) * out + gate * context)

        return enhanced, weights.squeeze(-1) if return_attention else None 

    def _strided_blocks(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """ Reshape [B, L, F] into [B * n_blocks, W, F] using trailing full blocks"""
        b, length, f = x.shape 
        w = self.short_window
        n_blocks = length // w
        if n_blocks < 1:
            raise ValueError(
                f"sequence_length = {length} is shorter than short_window = {w}; "
                "cannot form a single local block."
            )

        # Use the most recent n_blocks * w bars so any remainder is dropped from 
        # the far past rather than the recent past
        trimmed = x[:, length - n_blocks * w:, :]
        return trimmed.reshape(b * n_blocks, w, f), n_blocks

    # API 
    def forward(
        self,
        x: torch.Tensor,
        return_sequences: bool = False,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Encode ``x`` of shape [batch, seq_len, input_dim].
        
        With ``return_sequences = True`` returns the encoded sequence handed to 
        the Transformer:
        
        * ``strided`` -> [batch, seq_len // short_window, hidden_dim]
        * ``full`` -> [batch, seq_len, hidden_dim]
        * ``truncate`` -> [batch, short_window, hidden_dim]
        
        Otherwise returns standalone positions in [-1, 1] of shape [batch]
        """

        if self.mode == "strided":
            blocks, n_blocks = self._strided_blocks(x)
            encoded, attn = self._encode_window(blocks, return_attention_weights)
            # One embedding per block: the final timestep of each local window 
            block_embeddings = encoded[:, -1, :]
            sequence = block_embeddings.view(x.shape[0], n_blocks, self.hidden_dim)
            if attn is not None:
                attn = attn.view(x.shape[0], n_blocks, self.short_window)
        elif self.mode == "full":
            sequence, attn = self._encode_window(x, return_attention_weights)
        elif self.mode == "truncate":
            sequence, attn = self._encode_window(
                x[:, -self.short_window:, :], return_attention_weights
            )
        else:
            raise ValueError(f"Unknown encoder mode: {self.mode!r}")

        if return_sequences:
            return sequence, attn

        position = self.tanh(self.head(sequence[:, -1, :])).squeeze(-1)
        return position, attn

class LSTMMomentum(LSTMMomentumEncoder):
    """ Vanilla LSTM filter (no attention path)."""

    def __init__(self, *args, **kwargs):
        kwargs["use_attention_path"] = False
        super().__init__(*args, **kwargs)

class LSTMMomentumDualPath(LSTMMomentumEncoder):
    """ LSTM filter with the lightweight attention path enabled"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_attention_path", True)
        super().__init__(*args, **kwargs)

def get_lstm_encoder(config, use_attention: bool = False) -> LSTMMomentumEncoder:
    """ Build an encoder from a ``ModelConfig``.
    
    Unlike the previous factories this passes *every* configured value through,
    so ``lstm_num_layers``, ``lstm_dropout`` and ``short_window`` actually reach
    the module
    """

    return LSTMMomentumEncoder(
        input_dim = config.input_dim,
        hidden_dim = config.hidden_dim,
        num_layers = config.lstm_num_layers,
        dropout = config.lstm_dropout,
        short_window = config.short_window,
        use_attention_path = use_attention,
        mode = config.lstm_transformer_mode,
    )
