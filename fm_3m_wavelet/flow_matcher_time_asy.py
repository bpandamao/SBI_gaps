import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Callable, List
from residual_wavelet_try_asycnn_dilated3339 import create_signal_cf_model

# from residual_wavelet_try_asycnn_dilated33311 import create_signal_cf_model

# This part is unchanged
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

# --- Corrected Class ---
class ContinuousFlowMatcherTime(nn.Module):
    def __init__(
        self,
        param_dim: int = 3,
        signal_embedding_dim: int = 128, # The dimension of the feature vector extracted from the signal
        signal_input_dim: Optional[int] = None,  # For info only, not used by model
    ):
        super().__init__()
        
        # Create flow matching model with asymmetric kernels and dilation
        # signal_embedding_dim: dimension of the context vector from signal embedding network
        self.flow_model = create_signal_cf_model(
            signal_embedding_dim=signal_embedding_dim,
            posterior_kwargs={
                "output_dim": param_dim,
                # These hidden dimensions are for the final ResidualNet that combines parameter and signal info
                "hidden_dims": (1024,1024,512,256,128),#512, 256, 128, 64),
                "svd": { "size": 1024 },
                 # These hidden dimensions are for the network that embeds the parameters (theta)
                "param_hidden_dims": (64,128,256,512),
                "param_embedding_dim": 512,#256
            },
            activation=nn.GELU(),
            # batch_norm=True,
            dropout=0.4, #0.4
        )
          
        # Store dimensions for model_utils.py or other analysis
        self.param_dim = param_dim
        # The old `hidden_dim` is now `signal_embedding_dim`
        self.hidden_dim = signal_embedding_dim 
        self.signal_input_dim = signal_input_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, signal: Optional[torch.Tensor] = None):
        # Ensure proper tensor dimensions
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        return self.flow_model(t, x, signal)
