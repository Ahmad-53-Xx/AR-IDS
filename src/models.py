import torch
import torch.nn as nn

class TrilayeredIDSNetwork(nn.Module):
    """
    A deep, narrow network structure optimized for learning hierarchical 
    relationships in network flow features.
    """
    def __init__(self, input_dim):
        super(TrilayeredIDSNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)


class WideIDSNetwork(nn.Module):
    """
    A shallow, wide network structure. Fewer layers, but massive parallel
    processing capability to catch broad, independent correlation features.
    """
    def __init__(self, input_dim):
        super(WideIDSNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)
