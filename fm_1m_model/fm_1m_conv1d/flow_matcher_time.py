import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Callable, List
from residual_net import create_signal_cf_model

def timestep_embedding(timesteps, dim, max_period=10000):
    """Time embedding function from original implementation"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t):
        return self.net(timestep_embedding(t, self.hidden_dim))

class ContinuousFlowMatcherTime(nn.Module):
    def __init__(
        self,
        param_dim: int = 3,
        hidden_dim: int = 256,
        signal_input_dim: Optional[int] = None,  # Updated to match time domain signal length
    ):
        super().__init__()
        
        # Create signal flow model using create_signal_cf_model
        self.flow_model = create_signal_cf_model(
            signal_dim=signal_input_dim,
            hidden_dims=hidden_dim,
            posterior_kwargs={
                "output_dim": param_dim,
                "hidden_dims": (512,256, 128, 64),
                "param_hidden_dims": (64, 128, 256),
                "param_embedding_dim": 256,
            },
            activation=nn.GELU(),
            batch_norm=True,
            dropout=0.3,
        )
          
        # Store dimensions for model_utils.py
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.signal_input_dim = signal_input_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, signal: Optional[torch.Tensor] = None):
        # Ensure proper tensor dimensions
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        return self.flow_model(t, x, signal) 