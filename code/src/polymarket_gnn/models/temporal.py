"""Local Temporal GRU baseline model."""

import torch
import torch.nn as nn

class LocalGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        out, hn = self.gru(x)
        
        # Take the last time step for forecasting
        last_out = out[:, -1, :]
        
        # Predict target
        prediction = self.fc(last_out)
        return prediction.squeeze(-1)
