"""
PyTorch Datasets for Momentum Transformer

Rolling window dataset for time series data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset"""
    sequence_length: int = 252  # Full window (1 year)
    prediction_horizon: int = 1  # Predict next day
    stride: int = 1  # Step size between windows
    normalize: bool = True
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'


class RollingWindowDataset(Dataset):
    """
    Rolling window dataset for time series
    
    Creates overlapping windows of historical data for training
    
    Example:
        Data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Sequence length: 5
        Prediction horizon: 1
        Stride: 1
        
        Windows:
        [0,1,2,3,4] -> predict 5
        [1,2,3,4,5] -> predict 6
        [2,3,4,5,6] -> predict 7
        ...
    """
    def __init__(self,
                 features: np.ndarray,
                 returns: np.ndarray,
                 sequence_length: int = 252,
                 prediction_horizon: int = 1,
                 stride: int = 1):
        """
        Args:
            features: [T, F] - feature matrix (T timesteps, F features)
            returns: [T] - return series for prediction
            sequence_length: Length of input sequence
            prediction_horizon: How many steps ahead to predict
            stride: Step size between windows
        """
        self.features = features
        self.returns = returns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        # Calculate valid windows
        self.num_samples = (len(features) - sequence_length - prediction_horizon + 1) // stride
        
        if self.num_samples <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}. "
                           f"Need at least {sequence_length + prediction_horizon} timesteps, "
                           f"got {len(features)}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            x: [sequence_length, num_features] - input sequence
            y: scalar - return to predict
        """
        # Calculate start index
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1
        
        # Extract window
        x = self.features[start_idx:end_idx]
        y = self.returns[target_idx]
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y


class NormalizedRollingWindowDataset(Dataset):
    """
    Rolling window dataset with normalization
    
    Normalizes each window independently (rolling normalization)
    """
    def __init__(self,
                 features: np.ndarray,
                 returns: np.ndarray,
                 sequence_length: int = 252,
                 prediction_horizon: int = 1,
                 stride: int = 1,
                 normalization_method: str = 'zscore'):
        """
        Args:
            features: [T, F] - feature matrix
            returns: [T] - return series
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            stride: Step size
            normalization_method: 'zscore', 'minmax', or 'robust'
        """
        self.features = features
        self.returns = returns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.normalization_method = normalization_method
        
        self.num_samples = (len(features) - sequence_length - prediction_horizon + 1) // stride
        
        if self.num_samples <= 0:
            raise ValueError("Not enough data for given parameters")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized sample"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1
        
        # Extract window
        x = self.features[start_idx:end_idx].copy()
        y = self.returns[target_idx]
        
        # Normalize window
        x = self._normalize(x)
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize window"""
        if self.normalization_method == 'zscore':
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True) + 1e-8
            return (x - mean) / std
        
        elif self.normalization_method == 'minmax':
            min_val = np.min(x, axis=0, keepdims=True)
            max_val = np.max(x, axis=0, keepdims=True)
            return (x - min_val) / (max_val - min_val + 1e-8)
        
        elif self.normalization_method == 'robust':
            median = np.median(x, axis=0, keepdims=True)
            q75 = np.percentile(x, 75, axis=0, keepdims=True)
            q25 = np.percentile(x, 25, axis=0, keepdims=True)
            iqr = q75 - q25
            return (x - median) / (iqr + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")


class MultiHorizonDataset(Dataset):
    """
    Dataset for multi-step ahead prediction
    
    Used with decoder models that predict multiple future steps
    """
    def __init__(self,
                 features: np.ndarray,
                 returns: np.ndarray,
                 sequence_length: int = 252,
                 forecast_horizon: int = 5,
                 stride: int = 1):
        """
        Args:
            features: [T, F] - feature matrix
            returns: [T] - return series
            sequence_length: Length of input sequence
            forecast_horizon: Number of steps to predict
            stride: Step size
        """
        self.features = features
        self.returns = returns
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        self.num_samples = (len(features) - sequence_length - forecast_horizon + 1) // stride
        
        if self.num_samples <= 0:
            raise ValueError("Not enough data for given parameters")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample with multi-step targets
        
        Returns:
            x: [sequence_length, num_features] - input sequence
            y: [forecast_horizon] - returns to predict
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        target_start_idx = end_idx
        target_end_idx = target_start_idx + self.forecast_horizon
        
        # Extract window and targets
        x = self.features[start_idx:end_idx]
        y = self.returns[target_start_idx:target_end_idx]
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y


def create_train_val_test_split(features: np.ndarray,
                                returns: np.ndarray,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                sequence_length: int = 252,
                                **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/val/test split for time series
    
    Uses sequential split (no shuffling) to avoid lookahead bias
    
    Args:
        features: [T, F] - feature matrix
        returns: [T] - return series
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        sequence_length: Sequence length for dataset
        **kwargs: Additional arguments for dataset
        
    Returns:
        (train_dataset, val_dataset, test_dataset) tuple
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"
    
    total_length = len(features)
    
    # Calculate split points
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    
    # Ensure we have enough data for each split
    min_length = sequence_length + 10
    if train_end < min_length:
        raise ValueError(f"Training set too small. Need at least {min_length} samples.")
    
    # Split data
    train_features = features[:train_end]
    train_returns = returns[:train_end]
    
    val_features = features[train_end:val_end]
    val_returns = returns[train_end:val_end]
    
    test_features = features[val_end:]
    test_returns = returns[val_end:]
    
    # Create datasets
    train_dataset = RollingWindowDataset(
        train_features, train_returns, sequence_length, **kwargs
    )
    val_dataset = RollingWindowDataset(
        val_features, val_returns, sequence_length, **kwargs
    )
    test_dataset = RollingWindowDataset(
        test_features, test_returns, sequence_length, **kwargs
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Optional[Dataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       shuffle_train: bool = True) -> Tuple[DataLoader, ...]:
    """
    Create DataLoaders from datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


def load_data_from_csv(filepath: str,
                       feature_cols: List[str],
                       return_col: str = 'return') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        feature_cols: List of feature column names
        return_col: Name of return column
        
    Returns:
        (features, returns) tuple
    """
    df = pd.read_csv(filepath)
    
    # Extract features and returns
    features = df[feature_cols].values
    returns = df[return_col].values
    
    return features, returns


if __name__ == "__main__":
    print("Testing Dataset Module\n")
    
    # Create synthetic data
    np.random.seed(42)
    
    # 1000 timesteps, 10 features
    T = 1000
    F = 10
    
    features = np.random.randn(T, F)
    returns = np.random.randn(T) * 0.02
    
    print("Synthetic data:")
    print(f"Features shape: {features.shape}")
    print(f"Returns shape: {returns.shape}")
    
    # Test RollingWindowDataset
    print("\n" + "=" * 60)
    print("Testing RollingWindowDataset")
    print("=" * 60)
    
    dataset = RollingWindowDataset(
        features=features,
        returns=returns,
        sequence_length=252,
        prediction_horizon=1,
        stride=1
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    x, y = dataset[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")
    print(f"Sample values: y={y.item():.4f}")
    
    # Test NormalizedRollingWindowDataset
    print("\n" + "=" * 60)
    print("Testing NormalizedRollingWindowDataset")
    print("=" * 60)
    
    norm_dataset = NormalizedRollingWindowDataset(
        features=features,
        returns=returns,
        sequence_length=252,
        prediction_horizon=1,
        stride=1,
        normalization_method='zscore'
    )
    
    x_norm, y_norm = norm_dataset[0]
    print(f"Normalized sample shape: x={x_norm.shape}, y={y_norm.shape}")
    print(f"Normalized sample stats:")
    print(f"  Mean: {x_norm.mean():.4f}")
    print(f"  Std: {x_norm.std():.4f}")
    
    # Test train/val/test split
    print("\n" + "=" * 60)
    print("Testing Train/Val/Test Split")
    print("=" * 60)
    
    train_ds, val_ds, test_ds = create_train_val_test_split(
        features=features,
        returns=returns,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sequence_length=252
    )
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")
    
    # Test dataloaders
    print("\n" + "=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=32,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch_x.shape}")  # [batch, seq_len, features]
    print(f"  y: {batch_y.shape}")  # [batch]
    
    # Test MultiHorizonDataset
    print("\n" + "=" * 60)
    print("Testing MultiHorizonDataset")
    print("=" * 60)
    
    multi_dataset = MultiHorizonDataset(
        features=features,
        returns=returns,
        sequence_length=252,
        forecast_horizon=5,
        stride=1
    )
    
    print(f"Dataset length: {len(multi_dataset)}")
    
    x_multi, y_multi = multi_dataset[0]
    print(f"Sample shape: x={x_multi.shape}, y={y_multi.shape}")
    print(f"Target values: {y_multi.numpy()}")
    
    print("\n✅ Dataset module working correctly!")