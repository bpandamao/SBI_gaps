"""
Residual Network implementation for the decoupled flow matching method.
This module provides the ResidualNet class used in the flow matching model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional, Union


class ResidualBlock(nn.Module):
    """
    A single residual block with optional batch normalization and dropout.
    """
    def __init__(
        self,
        features: int,
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


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

