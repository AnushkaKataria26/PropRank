import torch
import torch.nn as nn

class RankNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be strictly positive, got {input_dim}")
            
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # returns shape (batch, 1)
        score = self.mlp(x)
        return score
