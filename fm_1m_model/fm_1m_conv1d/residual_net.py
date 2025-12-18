import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional, Union
import numpy as np

class ResidualBlock(nn.Module):
    """
    A single residual block with optional batch normalization, dropout, and GLU-style conditioning.
    """
    def __init__(
        self,
        features: int,
        context_features: Optional[int] = None,
        activation: Callable = nn.SiLU(),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features) if batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(features, features),
            nn.BatchNorm1d(features) if batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        else:
            self.context_layer = None
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.layers(x)
        
        if context is not None and self.context_layer is not None:
            context_embedding = self.context_layer(context)
            out = out * torch.sigmoid(context_embedding)
            
        return x + out

class ResidualNet(nn.Module):
    """
    Multi-block residual network with dimensionality reduction.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Union[int, Tuple[int, ...]],
        out_features: int,
        num_blocks: int = 2,
        activation: Callable = nn.SiLU(),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        first_hidden = hidden_features[0] if isinstance(hidden_features, tuple) else hidden_features
        self.input_layer = nn.Linear(in_features, first_hidden)
        
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        dims = hidden_features if isinstance(hidden_features, tuple) else [hidden_features] * num_blocks
        
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(
                features=dims[i],
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            ))
            if i < num_blocks - 1 and dims[i] != dims[i + 1]:
                self.transitions.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                self.transitions.append(nn.Identity())
        
        last_hidden = dims[-1]
        self.output_layer = nn.Linear(last_hidden, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block, transition in zip(self.blocks, self.transitions):
            x = block(x)
            x = transition(x)
        return self.output_layer(x)

class SignalConvEmbedding(nn.Module):
    """
    A 1D Convolutional Neural Network to embed the time-series signal.
    This network processes the signal through several convolutional and pooling
    layers to extract features and reduce its length efficiently.
    """
    def __init__(self, in_channels: int, output_embedding_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Each block consists of Conv1d, GELU activation, and MaxPool1d
            # Block 1
             nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Block 2: stride=6
            nn.Conv1d(64, 128, kernel_size=15, stride=6, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Block 3: stride=8
            nn.Conv1d(128, 256, kernel_size=17, stride=8, padding=8),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            # Block 4: stride=10
            nn.Conv1d(256, 512, kernel_size=21, stride=10, padding=10),
            nn.BatchNorm1d(512),
            nn.GELU()
        )
        
        # A final pooling layer and a linear layer to get to the desired embedding dimension
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(512, output_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, in_channels, signal_length]
        x = self.conv_layers(x)
        # Apply adaptive pooling to get a fixed-size output regardless of input length variations
        x = self.final_pool(x)
        # Flatten and project to the final embedding dimension
        x = x.view(x.size(0), -1)
        x = self.final_proj(x)
        return x

class SignalContinuousFlowModel(nn.Module):
    """
    Signal processing model for time domain signals (single channel).
    """
    def __init__(
        self,
        continuous_flow: nn.Module,
        param_embedding_net: nn.Module,
        signal_dim: int,
        hidden_dims: Tuple[int, ...],
        svd: dict, # Note: SVD dict is no longer used by the new embedding but kept for API consistency
        activation: Callable = nn.SiLU(),
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.signal_embedding = create_signal_embedding_net(
            signal_dim=signal_dim,
            output_dim=hidden_dims,
            hidden_dims=hidden_dims,
            svd=svd,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        
        self.param_embedding_net = param_embedding_net
        self.continuous_flow = continuous_flow

    def forward(self, t: torch.Tensor, theta: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0: t = t.view(1, 1)
        elif t.dim() == 1: t = t.unsqueeze(1)
        if theta.dim() == 1: theta = theta.unsqueeze(0)
            
        if t.size(0) == 1 and theta.size(0) > 1: t = t.expand(theta.size(0), -1)
        elif theta.size(0) == 1 and t.size(0) > 1: theta = theta.expand(t.size(0), -1)
            
        if signal.dim() == 2: x = signal.unsqueeze(1)
        else: x = signal
        
        signal_embedding = self.signal_embedding(x)
        param_input = torch.cat([t, theta], dim=1)
        param_embedding = self.param_embedding_net(param_input)
        
        if param_embedding.size(0) == 1 and signal_embedding.size(0) > 1:
            param_embedding = param_embedding.expand(signal_embedding.size(0), -1)
        elif signal_embedding.size(0) == 1 and param_embedding.size(0) > 1:
            signal_embedding = signal_embedding.expand(param_embedding.size(0), -1)
        
        combined = torch.cat([param_embedding, signal_embedding], dim=1)
        return self.continuous_flow(combined)

def create_signal_embedding_net(
    signal_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...],
    svd: dict, # No longer used, but kept for API consistency
    activation: Callable = nn.SiLU(),
    dropout: float = 0.0,
    batch_norm: bool = True,
):
    """Creates a signal embedding network using a 1D CNN."""
    # The new CNN front-end for feature extraction.
    # It takes 1 input channel (the raw signal) and produces 'output_dim' features.
    conv_embedding = SignalConvEmbedding(in_channels=1, output_embedding_dim=output_dim)
    
    # The CNN output can be directly used as the embedding.
    # Alternatively, you could add a ResidualNet here for further processing if needed,
    # but the CNN is already a very powerful feature extractor.
    return conv_embedding

def create_signal_cf_model(
    signal_dim: int,
    hidden_dims: Tuple[int, ...],
    posterior_kwargs: dict,
    activation: Callable = nn.SiLU(),
    batch_norm: bool = True,
    dropout: float = 0.0,
):
    """Creates a SignalContinuousFlowModel instance for time domain signals.
    (This function remains unchanged)
    """
    param_hidden_dims = posterior_kwargs.get("param_hidden_dims", [128, 128])
    param_embedding_net = ResidualNet(
        in_features=1 + posterior_kwargs["output_dim"],
        hidden_features=param_hidden_dims,
        out_features=posterior_kwargs.get("param_embedding_dim", 128),
        num_blocks=len(param_hidden_dims),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    
    # The input to the flow network comes from the CNN embedding
    # The in_features for the continuous_flow is the size of the parameter embedding
    # plus the size of the signal embedding, which is now hidden_dims[-1].
    continuous_flow = ResidualNet(
        in_features=posterior_kwargs.get("param_embedding_dim", 128) + hidden_dims,
        hidden_features=posterior_kwargs["hidden_dims"],
        out_features=posterior_kwargs["output_dim"],
        num_blocks=len(posterior_kwargs["hidden_dims"]),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    
    model = SignalContinuousFlowModel(
        continuous_flow=continuous_flow,
        param_embedding_net=param_embedding_net,
        signal_dim=signal_dim,
        hidden_dims=hidden_dims,
        svd=posterior_kwargs.get("svd", {"size": 256}),
        activation=activation,
        batch_norm=batch_norm,
        dropout=dropout,
    )
    
    return model