""" PyTorch Datasets for Momentum Transformer

* **Windows are checked for contiguity.** ``MultiAssetDataset.__getitem__``
    sliced positionally (``df.iloc[start:end]``), so after per-symbol NaN drops
    and split-bar deletions a "252-bar window" could silently span a gap of days 
    or weeks -- while the Transformer's positional encoding asserted the bars 
    were adjacent. Windows that are not contiguous on the trading calendar are 
    now rejected, and the rejection count is reported.
* **Samples carry their target timestamp.** ``index_frame`` exposes
    ``(symbol, timestamp)`` per sample, so predicitons can be joined to returns
    **by timestamp** rather than by position. 
* **Samples carry an ex-ante volatility** for the volatility-targeting layer
"""

from __future__ import annotations 

from dataclasses import dataclass 
from typing import Dict, List, Optional, Sequence, Tuple 

import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import DataLoader, Dataset 

METADATA_COLUMNS = {"symbol", "timestamp", "target", "ex_ante_vol"}

@dataclass 
class DatasetConfig:
    sequence_length: int = 252 # bars; see Models.config.ModelConfig
    stride: int = 1
    require_contiguous: bool = True 

class MultiAssetWindowDataset(Dataset):
    """ Rolling windows across many symbols, keyed by target timestamp
    
    ``features_df`` must contain ``symbol``, ``timestamp``, ``target`` and,
    optionally, ``ex_ante_vol``; every other column is treated as a feature
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        sequence_length: int = 252,
        feature_columns: Optional[Sequence[str]] = None,
        calendar: Optional[pd.DatetimeIndex] = None, 
        require_contiguous: bool = True,
        stride: int = 1,
        verbose: bool = True,
    ):
        if "target" not in features_df.columns:
            raise KeyError("features_df must contain a 'target' columns")

        self.sequence_length = sequence_length
        self.feature_columns: List[str] = list(
            feature_columns
            if feature_columns is not None
            else [c for c in features_df.columns if c not in METADATA_COLUMNS]
        )

        self._arrays: Dict[str, np.ndarray] = {}
        self._targets: Dict[str, np.ndarray] = {}
        self._vols: Dict[str, np.ndarray] = {}
        self._timestamps: Dict[str, pd.DatetimeIndex] = {}
        self.samples: List[Tuple[str, int]] = []

        n_rejected = 0
        calendar_pos = (
            pd.Series(np.arange(len(calendar)), index = pd.DatetimeIndex(calendar))
            if calendar is not None
            else None
        )

        for symbol_value, group in features_df.groupby("symbol", sort = True):
            symbol = str(symbol_value)
            group = group.sort_values("timestamp")
            if len(group) < sequence_length:
                continue 

            self._arrays[symbol] = group[self.feature_columns].to_numpy(dtype = np.float32)
            self._targets[symbol] = group["target"].to_numpy(dtype = np.float32)
            self._vols[symbol] = (
                group["ex_ante_vol"].to_numpy(dtype = np.float32)
                if "ex_ante_vol" in group.columns
                else np.zeros(len(group), dtype = np.float32)
            )

            timestamps = pd.DatetimeIndex(group["timestamp"])
            self._timestamps[symbol] = timestamps 

            positions = (
                calendar_pos.reindex(timestamps).to_numpy()
                if calendar_pos is not None
                else np.arange(len(timestamps))
            )

            for end in range(sequence_length - 1, len(group), stride):
                start = end - sequence_length + 1
                if require_contiguous and calendar_pos is not None:
                    window = positions[start:end + 1]
                    if np.isnan(window).any() or not np.all(np.diff(window) == 1):
                        n_rejected += 1
                        continue 
                self.samples.append((symbol, end))

        self.n_rejected = n_rejected 

        if verbose:
            print(
                f" Dataset: {len(self.samples):,} windows from "
                f"{len(self._arrays)} symbols "
                f"({n_rejected:,} rejected as non-contiguous)"
            )

    ### Introspection ###
    @property
    def index_frame(self) -> pd.DataFrame:
        """ ``(symbol, timestamp)`` for every sample, in dataset order
        
        Predictions are joined back to returns on these keys, never on 
        positional offsets.
        """

        return pd.DataFrame(
            {
                "symbol": [s for s, _ in self.samples],
                "timestamp": [self._timestamps[s][e] for s, e in self.samples],
            }
        )

    @property
    def group_ids(self) -> List[str]:
        """ Per-sample symbol, for ``SequentialBlockSampler``"""
        return [s for s, _ in self.samples]

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def targets_array(self) -> np.ndarray:
        return np.array([self._targets[s][e] for s, e in self.samples], dtype = np.float32)

    ### torch api ###
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        symbol, end = self.samples[idx]
        start = end - self.sequence_length + 1 

        x = torch.from_numpy(self._arrays[symbol][start:end + 1])
        y = torch.tensor(self._targets[symbol][end], dtype = torch.float32)
        vol = torch.tensor(self._vols[symbol][end], dtype = torch.float32)
        return x, y, vol 

class RollingWindowDataset(Dataset):
    """Single-series rolling windows, used by walk-forward analysis.
    
    ``target_index`` records which row of the source series each sample 
    predicts, so callers never have to infer the offset
    """

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        sequence_length: int = 252,
        stride: int = 1,
        timestamps: Optional[Sequence[object]] = None,
    ):
        features = np.asarray(features, dtype = np.float32)
        returns = np.asarray(returns, dtype = np.float32)

        if len(features) != len(returns):
            raise ValueError(
                f"features ({len(features)}) and returns ({len(returns)}) must align"
            )
        if len(features) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} rows, got {len(features)}"
            )

        self.features = features
        self.returns = returns 
        self.sequence_length = sequence_length 
        self.timestamps: Optional[pd.DatetimeIndex] = (
            pd.DatetimeIndex(list(timestamps)) if timestamps is not None else None
        )

        # Window [start .. end] predicts the target AT end (see 
        # Utils.Market_data.tradeable_returns for why it is not end + 1).
        self.target_index = np.arange(sequence_length - 1, len(features), stride)

    def __len__(self) -> int:
        return len(self.target_index)

    def __getitem__(self, idx: int):
        end = int(self.target_index[idx])
        start = end - self.sequence_length + 1
        x = torch.from_numpy(self.features[start:end + 1])
        y = torch.tensor(self.returns[end], dtype = torch.float32)
        return x, y 

    def target_timestamps(self) -> Optional[pd.DatetimeIndex]:
        if self.timestamps is None:
            return None 
        return self.timestamps[self.target_index]

def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, ...]:
    train_loader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = shuffle_train,
        num_workers = num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers
    )
    if test_dataset is None:
        return train_loader, val_loader 

    test_loader = DataLoader(
        test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers
    )
    return train_loader, val_loader, test_loader
