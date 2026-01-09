"""
Complete Example: Using the Momentum Transformer System

This script demonstrates:
1. Creating different model architectures
2. Training with Sharpe ratio loss
3. Using the ensemble with dynamic weighting (Strategy 2)
4. Using online performance monitoring (Strategy 4)
5. Combining both strategies
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import our modules
from Models.config import get_production_config
from Models.Momentum_transformer import (
    get_momentum_transformer,
    MomentumTransformerSimple,
    MomentumTransformerDualPath
)
from Models.Ensemble_model import get_ensemble_model
from Models.Online_Selector import OnlineModelSelector, HybridSelector
from Utils.Losses import sharpe_ratio_loss, get_loss_function


def generate_synthetic_data(num_samples=1000, seq_len=252, num_features=32):
    """
    Generate synthetic data for demonstration
    In production, replace with your actual feature engineering
    """
    X = torch.randn(num_samples, seq_len, num_features)
    # Synthetic returns (with some signal)
    y = torch.randn(num_samples) * 0.01
    
    return X, y


def create_dataloaders(X, y, train_ratio=0.7, val_ratio=0.15, batch_size=128):
    """Split data and create dataloaders"""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Split
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
    """
    Train model with Sharpe ratio loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_sharpe = -np.inf
    patience_counter = 0
    early_stopping_patience = 15
    
    print(f"Training {model.__class__.__name__}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (handle different model signatures)
            if isinstance(model, (MomentumTransformerSimple, MomentumTransformerDualPath)):
                if isinstance(model, MomentumTransformerDualPath):
                    positions, _ = model(X_batch, return_attention=False)
                else:
                    positions = model(X_batch)
            else:  # Ensemble or other
                positions, _ = model(X_batch, return_components=False)
            
            # Sharpe loss
            loss = sharpe_ratio_loss(positions, y_batch)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_positions = []
        val_returns = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if isinstance(model, (MomentumTransformerSimple, MomentumTransformerDualPath)):
                    if isinstance(model, MomentumTransformerDualPath):
                        pos, _ = model(X_batch, return_attention=False)
                    else:
                        pos = model(X_batch)
                else:
                    pos, _ = model(X_batch, return_components=False)
                
                val_positions.append(pos)
                val_returns.append(y_batch)
        
        val_positions = torch.cat(val_positions)
        val_returns = torch.cat(val_returns)
        val_sharpe = -sharpe_ratio_loss(val_positions, val_returns).item()
        
        # Learning rate scheduling
        scheduler.step(val_sharpe)
        
        # Early stopping
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={np.mean(train_losses):.4f}, Val Sharpe={val_sharpe:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
    print(f"Best validation Sharpe: {best_val_sharpe:.4f}\n")
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_positions = []
    all_returns = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if isinstance(model, (MomentumTransformerSimple, MomentumTransformerDualPath)):
                if isinstance(model, MomentumTransformerDualPath):
                    positions, _ = model(X_batch, return_attention=False)
                else:
                    positions = model(X_batch)
            else:
                positions, _ = model(X_batch, return_components=False)
            
            all_positions.append(positions)
            all_returns.append(y_batch)
    
    all_positions = torch.cat(all_positions)
    all_returns = torch.cat(all_returns)
    
    # Calculate metrics
    pnl = all_positions * all_returns
    
    sharpe = -sharpe_ratio_loss(all_positions, all_returns).item()
    mean_return = pnl.mean().item() * 252  # Annualized
    std_return = pnl.std().item() * np.sqrt(252)
    
    # Win rate
    win_rate = (pnl > 0).float().mean().item()
    
    return {
        'sharpe': sharpe,
        'return': mean_return,
        'volatility': std_return,
        'win_rate': win_rate
    }


def example_1_basic_models():
    """
    Example 1: Train and compare vanilla vs. attention-enhanced models
    """
    print("="*60)
    print("EXAMPLE 1: Basic Models Comparison")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(num_samples=2000, num_features=32)
    train_loader, val_loader, test_loader = create_dataloaders(X, y)
    
    # Configuration
    config = get_production_config()
    config.model.input_dim = 32
    
    # Create models
    vanilla_model = get_momentum_transformer(config.model, model_type='simple')
    enhanced_model = get_momentum_transformer(config.model, model_type='enhanced')
    
    # Train both
    vanilla_model = train_model(vanilla_model, train_loader, val_loader, num_epochs=30)
    enhanced_model = train_model(enhanced_model, train_loader, val_loader, num_epochs=30)
    
    # Evaluate
    vanilla_results = evaluate_model(vanilla_model, test_loader)
    enhanced_results = evaluate_model(enhanced_model, test_loader)
    
    print("\nTest Results:")
    print(f"Vanilla Model - Sharpe: {vanilla_results['sharpe']:.4f}, Return: {vanilla_results['return']:.4f}")
    print(f"Enhanced Model - Sharpe: {enhanced_results['sharpe']:.4f}, Return: {enhanced_results['return']:.4f}")
    print(f"Improvement: {(enhanced_results['sharpe'] - vanilla_results['sharpe']):.4f}\n")
    
    return vanilla_model, enhanced_model, test_loader


def example_2_ensemble():
    """
    Example 2: Ensemble with dynamic weighting (Strategy 2)
    """
    print("="*60)
    print("EXAMPLE 2: Ensemble with Dynamic Weighting")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(num_samples=2000, num_features=32)
    train_loader, val_loader, test_loader = create_dataloaders(X, y)
    
    # Configuration
    config = get_production_config()
    config.model.input_dim = 32
    
    # Create ensemble
    ensemble_model = get_ensemble_model(config)
    
    # Train
    ensemble_model = train_model(ensemble_model, train_loader, val_loader, num_epochs=40)
    
    # Evaluate with detailed analysis
    ensemble_model.eval()
    all_metadata = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            positions, metadata = ensemble_model(X_batch, return_components=True)
            all_metadata.append(metadata)
    
    # Analyze ensemble weights
    weight_stats = ensemble_model.get_weight_statistics()
    
    print("\nEnsemble Weight Statistics:")
    print(f"  Mean attention weight: {weight_stats['mean_attention_weight']:.3f}")
    print(f"  Attention model used: {weight_stats['attention_usage_pct']:.1f}%")
    print(f"  Vanilla model used: {weight_stats['vanilla_usage_pct']:.1f}%")
    
    results = evaluate_model(ensemble_model, test_loader)
    print(f"\nTest Sharpe: {results['sharpe']:.4f}\n")
    
    return ensemble_model, test_loader


def example_3_online_selector():
    """
    Example 3: Online performance monitoring (Strategy 4)
    """
    print("="*60)
    print("EXAMPLE 3: Online Performance Monitoring")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(num_samples=2000, num_features=32)
    train_loader, val_loader, test_loader = create_dataloaders(X, y)
    
    # Configuration
    config = get_production_config()
    config.model.input_dim = 32
    
    # Create and train both models
    vanilla_model = get_momentum_transformer(config.model, model_type='simple')
    attention_model = get_momentum_transformer(config.model, model_type='enhanced')
    
    vanilla_model = train_model(vanilla_model, train_loader, val_loader, num_epochs=30)
    attention_model = train_model(attention_model, train_loader, val_loader, num_epochs=30)
    
    # Create online selector
    selector = OnlineModelSelector(
        vanilla_model=vanilla_model,
        attention_model=attention_model,
        lookback_window=50,
        switch_threshold=0.2,
        min_observations=20
    )
    
    # Simulate trading with online monitoring
    print("\nSimulating online trading...")
    for i, (X_batch, y_batch) in enumerate(test_loader):
        # Get predictions
        positions, metadata = selector.predict(X_batch)
        
        # Simulate execution and get realized returns
        realized_returns = y_batch.numpy()
        
        # Update performance tracking
        perf_meta = selector.update_performance(realized_returns)
        
        if i % 5 == 0 and perf_meta.get('vanilla_sharpe'):
            print(f"Batch {i}: Using {metadata['model_used']}, "
                  f"V_Sharpe={perf_meta['vanilla_sharpe']:.3f}, "
                  f"A_Sharpe={perf_meta['attention_sharpe']:.3f}")
    
    # Final summary
    summary = selector.get_performance_summary()
    print("\nFinal Performance Summary:")
    print(f"  Current model: {summary['current_model']}")
    print(f"  Number of switches: {summary['num_switches']}")
    
    if 'vanilla' in summary:
        print(f"  Vanilla - Sharpe: {summary['vanilla']['sharpe_ratio']:.4f}")
    if 'attention' in summary:
        print(f"  Attention - Sharpe: {summary['attention']['sharpe_ratio']:.4f}")
    
    print()
    return selector


def example_4_hybrid():
    """
    Example 4: Hybrid approach combining ensemble and online monitoring
    """
    print("="*60)
    print("EXAMPLE 4: Hybrid (Ensemble + Online Monitoring)")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(num_samples=2000, num_features=32)
    train_loader, val_loader, test_loader = create_dataloaders(X, y)
    
    # Configuration
    config = get_production_config()
    config.model.input_dim = 32
    
    # Create ensemble
    ensemble_model = get_ensemble_model(config)
    
    # Create component models for monitoring
    vanilla_model = get_momentum_transformer(config.model, model_type='simple')
    attention_model = get_momentum_transformer(config.model, model_type='enhanced')
    
    # Train all models
    ensemble_model = train_model(ensemble_model, train_loader, val_loader, num_epochs=40)
    vanilla_model = train_model(vanilla_model, train_loader, val_loader, num_epochs=30)
    attention_model = train_model(attention_model, train_loader, val_loader, num_epochs=30)
    
    # Create hybrid selector
    hybrid = HybridSelector(
        ensemble_model=ensemble_model,
        vanilla_model=vanilla_model,
        attention_model=attention_model,
        lookback_window=50,
        switch_threshold=0.3,
        min_observations=30
    )
    
    # Test hybrid approach
    print("\nTesting hybrid approach...")
    for i, (X_batch, y_batch) in enumerate(test_loader):
        positions, metadata = hybrid(X_batch, return_all=True)
        
        # Update monitoring
        realized_returns = y_batch.numpy()
        hybrid.update_performance(realized_returns)
        
        if i % 5 == 0:
            print(f"Batch {i}: Ensemble weight={metadata['ensemble_mean_weight']:.3f}, "
                  f"Online recommends={metadata['online_recommended_model']}")
    
    # Get full summary
    summary = hybrid.get_full_summary()
    
    print("\nHybrid Summary:")
    print("Ensemble Weights:")
    for key, value in summary['ensemble_weights'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nOnline Performance:")
    if 'current_model' in summary['online_performance']:
        print(f"  Recommended model: {summary['online_performance']['current_model']}")
    
    print()
    return hybrid


def main():
    """
    Run all examples
    """
    print("\n" + "="*60)
    print("MOMENTUM TRANSFORMER - COMPLETE EXAMPLES")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    print("Running examples with synthetic data...")
    print("In production, replace with your actual feature engineering\n")
    
    try:
        # Example 1: Basic comparison
        vanilla, enhanced, test_loader = example_1_basic_models()
        
        # Example 2: Ensemble
        ensemble, _ = example_2_ensemble()
        
        # Example 3: Online selector
        selector = example_3_online_selector()
        
        # Example 4: Hybrid
        hybrid = example_4_hybrid()
        
        print("="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()