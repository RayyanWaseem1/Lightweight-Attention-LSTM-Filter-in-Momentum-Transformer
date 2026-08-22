""" Momentum Transformer: local LSTM filter + global Transformer attention.

Fixes from the previous version

* **The head ends in ``tanh``.** Every docstring claimed positions in 
    ``[-1,1]`` and every loss assumed it, but the head emitted an unbounded 
    real. Because the Sharpe objective is scale-invariant
    (``Sharpe(c * pnl) == Sharpe(pnl)``), position *magnitude* was completely
    unidentified by training -- the output scale was an accident of 
    initialization, the turnover penalty was meaningless, and downstream code
    patched it with ``min_signal`` thresholds, and ``std * 2`` rescaling. 
    
* **Volatility targeting is implemented**, using the ``target_volatility``
    field that previously sat unread in ``TrainingConfig``:
    ``position = tanh(z) * target_vol / ex_ante_vol``, capped at ``max_leverage``
    
* **A uniform ``(position, info)`` return signature`` for every model, removing 
    the five downstream ``out[0] if isinstance(out, tuple) else out`` special cases
    
* **The factories pass the whole config through.** They previously hardcoded
    ``lstm_num_layers = 2``, ``short_window = 63`` and ``feedforward_dim = hidden * 4``,
    so ``config.model.lstm_num_layers``, ``.short_window``, ``.feedforward_dim``
    and ``.lstm_dropout``, never reached the model -- and Ray Tune had searched
    over ``lstm_num_layers`` and ``lstm_dropout``, so those dimensions of the 
    hyperparameter search did nothing at all.
"""

from __future__ import annotations 

from typing import Dict, Literal, Optional, Tuple

import torch 
import torch.nn as nn 

from .LSTM import LSTMMomentumEncoder
from .Transformer_layers import TransformerEncoder 

ModelOutput = Tuple[torch.Tensor, Optional[Dict]]

def apply_volatility_target(
    raw_position: torch.Tensor,
    ex_ante_vol: Optional[torch.Tensor],
    target_volatility: float, 
    max_leverage: float = 3.0,
) -> torch.Tensor:
    """ Scale a bounded position by ``target_vol / ex_ante_vol``
    
    ``ex_ante_vol`` is an **annualized** volatility forecast per sample, formed
    from information available stricly before the bar being traded. The
    multiplier is capped so a near-zero volatility estimate cannot produce
    unbounded leverage.
    """

    if ex_ante_vol is None:
        return raw_position 

    vol = ex_ante_vol.reshape(-1).to(raw_position.dtype)
    multiplier = torch.clamp(target_volatility / (vol + 1e-6), max = max_leverage)
    return raw_position * multiplier 

class MomentumTransformer(nn.Module):
    """ LSTM local encoder feeding a Transformer global encoder"""

    def __init__(
        self, 
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
        use_positional_encoding: bool = True,
        use_causal_mask: bool = True,
        lstm_transformer_mode: Literal["strided", "full", "truncate"] = "strided",
        use_volatility_targeting: bool = True,
        target_volatility: float = 0.15,
        max_leverage: float = 3.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim 
        self.use_lstm_attention = use_lstm_attention 
        self.use_volatility_targeting = use_volatility_targeting
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage 

        self.lstm: LSTMMomentumEncoder = LSTMMomentumEncoder(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            num_layers = lstm_num_layers,
            dropout = lstm_dropout,
            short_window = short_window,
            use_attention_path = use_lstm_attention,
            mode = lstm_transformer_mode,
        )

        self.transformer: TransformerEncoder = TransformerEncoder(
            d_model = hidden_dim,
            num_layers = num_transformer_layers,
            num_heads = num_attention_heads,
            dim_feedforward = feedforward_dim,
            dropout = transformer_dropout,
            use_positional_encoding = use_positional_encoding,
            use_causal_mask = use_causal_mask,
        )

        self.prediction_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(), # the output IS a position
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        ex_ante_vol: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """ ``x``: [batch, seq_len, input_dim] -> positions [batch]"""
        lstm_out, lstm_attention = self.lstm(
            x, return_sequences = True, return_attention_weights = return_attention
        )
        # lstm_out: [batch, n_local_blocks | seq_len | short_window, hidden_dim]

        transformer_out, transformer_attention = self.transformer(
            lstm_out, return_attention = return_attention
        )

        final_repr = transformer_out[:, -1, :]
        raw_position = self.prediction_head(final_repr).squeeze(-1)

        position = raw_position
        if self.use_volatility_targeting:
            position = apply_volatility_target(
                raw_position, ex_ante_vol, self.target_volatility, self.max_leverage
            )

        info = None 
        if return_attention:
            info = {
                "lstm_attention": lstm_attention,
                "transformer_attention": transformer_attention,
                "raw_position": raw_position.detach(),
                "encoded_length": lstm_out.shape[1],
            }
        return position, info

class MomentumTransformerSimple(nn.Module):
    """ Vanilla arm: LSTM filter without the attention path"""

    def __init__(self, **kwargs):
        super().__init__()
        kwargs["use_lstm_attention"] = False 
        self.model: MomentumTransformer = MomentumTransformer(**kwargs)

    def forward(self, x, return_attention: bool = False, ex_ante_vol = None) -> ModelOutput:
        return self.model(x, return_attention = return_attention, ex_ante_vol = ex_ante_vol)

class MomentumTransformerDualPath(nn.Module):
    """Attention arm: identical capacity plus the LSTM attention/gate path"""

    def __init__(self, **kwargs):
        kwargs.setdefault("use_lstm_attention", True)
        super().__init__()
        self.model: MomentumTransformer = MomentumTransformer(**kwargs)

    def forward(self, x, return_attention: bool = False, ex_ante_vol = None) -> ModelOutput:
        return self.model(x, return_attention = return_attention, ex_ante_vol = ex_ante_vol)

def model_kwargs_from_config(config, training_config = None) -> Dict:
    """ Translate a ``ModelConfig`` into ``MomentumTransformer`` kwargs.
    
    Single place where config becomes constructor arguments, so no field can be 
    silently dropped on the way in 
    """

    kwargs = dict(
        input_dim = config.input_dim,
        hidden_dim = config.hidden_dim,
        lstm_num_layers = config.lstm_num_layers,
        lstm_dropout = config.lstm_dropout,
        short_window = config.short_window,
        num_transformer_layers = config.num_transformer_layers,
        num_attention_heads = config.num_attention_heads,
        transformer_dropout = config.transformer_dropout,
        feedforward_dim = config.feedforward_dim,
        use_positional_encoding = config.use_positional_encoding,
        use_causal_mask = config.use_causal_mask,
        lstm_transformer_mode = config.lstm_transformer_mode,
        use_volatility_targeting = config.use_volatility_targeting,
    )

    if training_config is not None:
        kwargs["target_volatility"] = training_config.target_volatility
    return kwargs 

def get_momentum_transformer(config, model_type: str = "enhanced", training_config = None):
    """ Factory: ``model_type`` in {"simple", "enhanced"}"""

    kwargs = model_kwargs_from_config(config, training_config)

    if model_type == "simple":
        return MomentumTransformerSimple(**kwargs)
    if model_type == "enhanced":
        kwargs["use_lstm_attention"] = config.use_lstm_attention
        return MomentumTransformerDualPath(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")

def count_parameters(model: nn.Module) -> int:
    """ Trainable parameter count (``numel``, not the previous ``num()`` typo)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
