import torch
import torch.nn as nn
import math
from typing import Tuple, Callable, Optional, Union
import numpy as np

#==============================================================================
# Generic Building Blocks (Unchanged)
#==============================================================================

class ResidualBlock(nn.Module):
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
    def __init__(
        self,
        in_features: int,
        hidden_features: Union[int, Tuple[int, ...]],
        out_features: int,
        num_blocks: int,
        activation: Callable = nn.SiLU(),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        
        # Ensure hidden_features is a list/tuple
        dims = hidden_features if isinstance(hidden_features, (list, tuple)) else [hidden_features] * num_blocks
        
        self.input_layer = nn.Linear(in_features, dims[0])
        
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(features=dims[i], activation=activation, dropout=dropout, batch_norm=batch_norm))
            if i < num_blocks - 1 and dims[i] != dims[i + 1]:
                self.transitions.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                self.transitions.append(nn.Identity())
        
        self.output_layer = nn.Linear(dims[-1], out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block, transition in zip(self.blocks, self.transitions):
            x = block(x)
            x = transition(x)
        return self.output_layer(x)

#==============================================================================
# Wavelet-based 2D Convolutional Feature Extractor (MODIFIED)
#==============================================================================

class Conv2dResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: Callable = nn.GELU()):
        super().__init__()
        
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.final_activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.main_path(x) + self.shortcut(x))


class SignalConvEmbedding(nn.Module):
    def __init__(self, output_embedding_dim: int):
        super().__init__()
        self.in_channels = 64
        self.initial_layer = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU(),
            # --- CHANGE 1: REMOVED MaxPool2d layer to preserve positional information ---
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)
        
        # --- CHANGE 2: REPLACED complex dual pooling with a single AdaptiveAvgPool2d ---
        # This is a less destructive way to flatten the feature map.
        self.summary_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- CHANGE 3: ADJUSTED in_features for the final projection layer ---
        # The previous dual pooling concatenated two 1024-dim vectors (2048).
        # The new single pooling produces one 1024-dim vector.
        self.final_proj = ResidualNet(
            in_features=512, # MODIFIED from 2048
            hidden_features=512,
            out_features=output_embedding_dim,
            num_blocks=1,
            dropout=0.4,
            activation=nn.GELU()
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Conv2dResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        out = self.initial_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.summary_pool(out)
        out = out.view(x.size(0), -1) 
        final_embedding = self.final_proj(out)
        return final_embedding

#==============================================================================
# Factory and Top-Level Model (Unchanged)
#==============================================================================

class SignalContinuousFlowModel(nn.Module):
    def __init__(self, continuous_flow: nn.Module, param_embedding_net: nn.Module, signal_embedding_net: nn.Module):
        super().__init__()
        self.signal_embedding = signal_embedding_net
        self.param_embedding_net = param_embedding_net
        self.continuous_flow = continuous_flow

    def forward(self, t: torch.Tensor, theta: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0: t = t.view(1, 1)
        elif t.dim() == 1: t = t.unsqueeze(1)
        if theta.dim() == 1: theta = theta.unsqueeze(0)
            
        if t.size(0) == 1 and theta.size(0) > 1: t = t.expand(theta.size(0), -1)
        elif theta.size(0) == 1 and t.size(0) > 1: theta = theta.expand(t.size(0), -1)
            
        signal_embedding = self.signal_embedding(signal)
        param_input = torch.cat([t, theta], dim=1)
        param_embedding = self.param_embedding_net(param_input)
        
        if param_embedding.size(0) != signal_embedding.size(0):
             signal_embedding = signal_embedding.expand(param_embedding.size(0), -1)
        
        combined = torch.cat([param_embedding, signal_embedding], dim=1)
        return self.continuous_flow(combined)


def create_signal_embedding_net(
    output_dim: int,
    activation: Callable = nn.ReLU(),
    dropout: float = 0.0,
    batch_norm: bool = True,
):
    conv_embedding = SignalConvEmbedding(
        output_embedding_dim=output_dim
    )
    return conv_embedding

def create_signal_cf_model(
    signal_embedding_dim: int,
    posterior_kwargs: dict,
    activation: Callable = nn.SiLU(),
    batch_norm: bool = True,
    dropout: float = 0.0,
):
    param_hidden_dims = posterior_kwargs.get("param_hidden_dims", (128, 128))
    param_embedding_dim = posterior_kwargs.get("param_embedding_dim", 128)

    param_embedding_net = ResidualNet(
        in_features=1 + posterior_kwargs["output_dim"],
        hidden_features=param_hidden_dims,
        out_features=param_embedding_dim,
        num_blocks=len(param_hidden_dims),
        activation=activation, dropout=dropout, batch_norm=batch_norm,
    )
    
    flow_hidden_dims = posterior_kwargs["hidden_dims"]
    continuous_flow = ResidualNet(
        in_features=param_embedding_dim + signal_embedding_dim,
        hidden_features=flow_hidden_dims,
        out_features=posterior_kwargs["output_dim"],
        num_blocks=len(flow_hidden_dims),
        activation=activation, dropout=dropout, batch_norm=batch_norm,
    )
    
    signal_embedding_net = create_signal_embedding_net(
        output_dim=signal_embedding_dim,
        activation=nn.GELU(),
        dropout=dropout, 
        batch_norm=batch_norm,
    )
    
    model = SignalContinuousFlowModel(
        continuous_flow=continuous_flow,
        param_embedding_net=param_embedding_net,
        signal_embedding_net=signal_embedding_net
    )
    
    return model
