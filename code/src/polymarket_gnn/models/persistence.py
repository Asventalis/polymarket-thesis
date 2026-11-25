"""Persistence baseline model using PyTorch and GPU support."""

import numpy as np
import torch

class PersistenceBaseline:
    """A simple baseline that predicts no change in logit price.
    
    Since target = logit(P_{t+h}) - logit(P_t), the persistence 
    prediction is simply 0.0 for all cases.
    """
    def __init__(self, device=None):
        self.device = device
        
    def fit(self, X, y=None):
        """Fit does nothing for persistence."""
        return self
        
    def predict(self, X):
        """Predict exactly 0.0 for all inputs.
        
        Args:
            X: Input features (numpy array or torch.Tensor)
            
        Returns:
            Zeros matching input format (torch.Tensor or numpy array)
        """
        if isinstance(X, torch.Tensor):
            device = self.device if self.device is not None else X.device
            return torch.zeros(X.shape[0], device=device, dtype=X.dtype)
        num_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.zeros(num_samples)

