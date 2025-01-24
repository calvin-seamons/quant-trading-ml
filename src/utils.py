import numpy as np
import torch
from scipy.stats import spearmanr
from typing import Dict
import pandas as pd
from typing import List


def calculate_financial_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Calculate financial-specific metrics
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for calculations
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    # Direction Accuracy (similar to classification accuracy for up/down movements)
    direction_correct = np.mean((y_true > 0) == (y_pred > 0))
    
    # Sharpe-like Ratio (using predictions as position sizes)
    strategy_returns = y_true * np.sign(y_pred)  # Long/short based on predictions
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-7) * np.sqrt(252)
    
    # Information Coefficient (Spearman rank correlation)
    ic = spearmanr(y_true, y_pred)[0]
    
    # RMSE scaled by volatility (similar to information ratio)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    vol_scaled_rmse = rmse / (np.std(y_true) + 1e-7)
    
    return {
        'direction_accuracy': float(direction_correct),
        'sharpe_ratio': float(sharpe),
        'information_coefficient': float(ic),
        'vol_scaled_rmse': float(vol_scaled_rmse),
        'rmse': float(rmse)
    }

def validate_temporal_split(train_indices, test_indices) -> bool:
    """
    Verify that train data strictly precedes test data to prevent lookahead bias
    
    Args:
        train_indices: Indices used for training
        test_indices: Indices used for testing
        
    Returns:
        bool: True if split is temporally valid
    """
    if len(train_indices) == 0 or len(test_indices) == 0:
        return False
        
    # Verify all training indices come before test indices
    last_train_idx = max(train_indices)
    first_test_idx = min(test_indices)
    
    return last_train_idx < first_test_idx

def check_forward_looking_features(features: pd.DataFrame) -> List[str]:
    """
    Check for potential forward-looking features by analyzing autocorrelation
    with future values
    
    Args:
        features: DataFrame of features
        
    Returns:
        List of potentially problematic features
    """
    suspicious_features = []
    
    for col in features.columns:
        if col == 'symbol':  # Skip symbol column
            continue
            
        if isinstance(features[col].iloc[0], (int, float)):
            # Check correlation with future values
            future_corr = features[col].corr(features[col].shift(-21))  # Using 21 days as default window
            if abs(future_corr) > 0.9:  # High correlation threshold
                suspicious_features.append({
                    'feature': col,
                    'future_correlation': future_corr
                })
    
    return suspicious_features