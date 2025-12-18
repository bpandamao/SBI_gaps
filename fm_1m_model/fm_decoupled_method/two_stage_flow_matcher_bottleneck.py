import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from residual_net import ResidualNet # Reuse your ResidualNet implementation
import math

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

class ContinuousFlowMatcherBottleneck(nn.Module):
    """
    Flow matching model conditioned on a 256-dim bottleneck vector
    AND 1-dim scale (log_std) vector.
    """
    def __init__(
        self,
        param_dim: int = 3,
        conditioning_dim: int = 257, # Changed from bottleneck_dim
        hidden_dim: int = 256,
        time_embed_dim: int = 64
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.conditioning_dim = conditioning_dim # Changed
        self.hidden_dim = hidden_dim
        
        # Time embedding network
        self.time_embed_net = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embed_dim = time_embed_dim
        
        # Bottleneck embedding network
        self.conditioning_embed_net = nn.Sequential( # Renamed
            nn.Linear(conditioning_dim, hidden_dim), # Changed
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Parameter embedding network
        self.param_embed_net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main flow model (a simple ResidualNet)
        # Input features:
        # 1. Parameter embedding (hidden_dim)
        # 2. Time embedding (hidden_dim)
        # 3. Conditioning embedding (hidden_dim) # Renamed
        self.flow_model = ResidualNet(
            in_features=hidden_dim * 3,
            hidden_features=(512, 1024, 512, 256),
            out_features=param_dim,
            num_blocks=4,
            activation=nn.GELU(),
            dropout=0.2,
            batch_norm=True
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor): # Renamed
        # x_t: [B, param_dim] (parameters at time t)
        # t: [B] (time)
        # condition: [B, conditioning_dim] (bottleneck + log_std)

        # Ensure t is 1D
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() > 1:
            t = t.squeeze()
            
        # 1. Embed time
        t_embedded = timestep_embedding(t, self.time_embed_dim)
        t_embedded = self.time_embed_net(t_embedded) # [B, hidden_dim]
        
        # 2. Embed parameters (x_t)
        x_t_embedded = self.param_embed_net(x_t) # [B, hidden_dim]
        
        # Get the batch size from x_t, which is the true batch size
        batch_size = x_t.shape[0]
        # The ODE solver passes t as a scalar, so t_embedded might be [1, hidden_dim].
        # If x_t is batched (e.g., [500, hidden_dim]), we must expand t_embedded
        # to match the batch size.
        if t_embedded.shape[0] == 1 and batch_size > 1:
            t_embedded = t_embedded.expand(batch_size, -1) # [B, hidden_dim]
        # --- END FIX ---
        
        # 3. Embed conditioning vector
        conditioning_embedded = self.conditioning_embed_net(condition) # [B, hidden_dim]
        
        # 4. Combine embeddings
        combined_embedding = torch.cat([x_t_embedded, t_embedded, conditioning_embedded], dim=1) # [B, hidden_dim * 3]
        
        # 5. Pass through flow model
        v_pred = self.flow_model(combined_embedding) # [B, param_dim]
            
        return v_pred