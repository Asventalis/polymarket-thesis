"""Tabular baseline models with PyTorch GPU acceleration."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
import lightgbm as lgb

class PyTorchRidgeBaseline:
    """Closed-form Ridge Regression computed entirely on GPU via PyTorch tensors."""
    def __init__(self, alpha: float = 1.0, device: str = None):
        self.alpha = alpha
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(dtype=torch.float32, device=self.device)

        X = torch.nan_to_num(X, 0.0)
        y = torch.nan_to_num(y, 0.0)

        # Add bias column
        N, D = X.shape
        X_b = torch.cat([X, torch.ones((N, 1), device=self.device, dtype=torch.float32)], dim=1)

        # Ridge regularized normal equations: (X_b^T X_b + alpha * I) w = X_b^T y
        reg = self.alpha * torch.eye(D + 1, device=self.device, dtype=torch.float32)
        reg[-1, -1] = 0.0  # Do not regularize bias term

        A = torch.matmul(X_b.T, X_b) + reg
        b = torch.matmul(X_b.T, y)

        sol = torch.linalg.solve(A, b)
        self.weights = sol[:D]
        self.bias = sol[-1]
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_t = X.to(dtype=torch.float32, device=self.device)

        X_t = torch.nan_to_num(X_t, 0.0)
        preds = torch.matmul(X_t, self.weights) + self.bias
        if isinstance(X, np.ndarray):
            return preds.cpu().numpy()
        return preds


class PyTorchTabularMLP(nn.Module):
    """PyTorch MLP Tabular baseline trained on GPU."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, device: str = None):
        super().__init__()
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)

    def fit(self, X, y, eval_set=None, epochs: int = 20, batch_size: int = 2048, lr: float = 1e-3):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)
            y = y.to(dtype=torch.float32, device=self.device)

        X = torch.nan_to_num(X, 0.0)
        y = torch.nan_to_num(y, 0.0)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss()

        self.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_t = X.to(dtype=torch.float32, device=self.device)
            X_t = torch.nan_to_num(X_t, 0.0)
            preds = self(X_t)
            if isinstance(X, np.ndarray):
                return preds.cpu().numpy()
            return preds


class RidgeBaseline:
    """Scikit-learn fallback Ridge Baseline."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        X = np.nan_to_num(X)
        return self.model.predict(X)


class LightGBMBaseline:
    """LightGBM Baseline."""
    def __init__(self, **params):
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1
        }
        default_params.update(params)
        self.params = default_params
        self.model = None
        
    def fit(self, X, y, eval_set=None):
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
        )
        return self
        
    def predict(self, X):
        return self.model.predict(X)

