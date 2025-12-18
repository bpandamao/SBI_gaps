"""
Masked Autoregressive Flow (MAF) model implementation for gravitational wave parameter estimation.
"""

import torch
import torch.nn as nn
from glasflow.nflows import transforms, distributions, flows

class ProgressiveContextNet(nn.Module):
    """
    The 'central power station' CNN that processes the raw signal into an embedding.
    This is only created ONCE.
    """
    def __init__(self, context_dim, output_embedding_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, stride=6, padding=7),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=17, stride=8, padding=8),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=21, stride=10, padding=10),
            nn.BatchNorm1d(512), nn.GELU()
        )
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(512, output_embedding_dim)

    def forward(self, context):
        """Transform context through the convolutional network."""
        # Context comes in as [batch_size, signal_length].
        # Conv1d expects [batch_size, in_channels, signal_length].
        x = context.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.final_proj(x)
        return x

def create_transform_block(param_dim, context_embedding_dim, hidden_dims, num_transform_blocks, use_batch_norm):
    """
    Creates the full sequence of transforms for the flow.
    Each transform in the sequence will share the same context embedding.
    """
    transform_list = []
    for hidden_dim in hidden_dims:
        transform_list.extend([
            transforms.MaskedAffineAutoregressiveTransform(
                features=param_dim,
                hidden_features=hidden_dim,
                context_features=context_embedding_dim,
                num_blocks=num_transform_blocks,
                use_residual_blocks=True,
                activation=torch.nn.GELU(),
                use_batch_norm=use_batch_norm,
                dropout_probability=0.0
            ),
            transforms.RandomPermutation(features=param_dim)
        ])
    # The last permutation is not strictly necessary but is common.
    return transforms.CompositeTransform(transform_list)

class ConditionalMAF(nn.Module):
    """
    The main, efficient model class.
    It contains the context embedder and the normalizing flow, ensuring the
    context is processed only once.
    """
    def __init__(self, param_dim, context_dim, hidden_dims, num_transform_blocks, use_batch_norm, context_embedding_dim):
        super().__init__()
        # The single context embedder (CNN)
        self.context_embedder = ProgressiveContextNet(
            context_dim=context_dim,
            output_embedding_dim=context_embedding_dim
        )
        
        # The sequence of transformations (the flow)
        transform = create_transform_block(
            param_dim=param_dim,
            context_embedding_dim=context_embedding_dim,
            hidden_dims=hidden_dims,
            num_transform_blocks=num_transform_blocks,
            use_batch_norm=use_batch_norm
        )
        
        # The base distribution
        base_distribution = distributions.StandardNormal(shape=[param_dim])
        
        # The complete flow object
        self.flow = flows.Flow(transform, base_distribution)

    def forward(self, inputs, context):
        """Defines the forward pass of the model."""
        # 1. Process context once to get the embedding
        embedding = self.context_embedder(context)
        # 2. Pass inputs and the shared embedding to the flow
        return self.flow.forward(inputs, context=embedding)

    def log_prob(self, inputs, context):
        """Calculates the log probability of the inputs given the context."""
        embedding = self.context_embedder(context)
        return self.flow.log_prob(inputs, context=embedding)

    def sample(self, num_samples, context):
        """Generates samples from the flow given the context."""
        embedding = self.context_embedder(context)
        return self.flow.sample(num_samples, context=embedding)

def create_maf_model(param_dim, context_dim, hidden_dims: list, num_transform_blocks=2, batch_norm=False, context_embedding_dim: int = 256):
    """
    Factory function to create the efficient conditional MAF model.
    """
    return ConditionalMAF(
        param_dim=param_dim,
        context_dim=context_dim,
        hidden_dims=hidden_dims,
        num_transform_blocks=num_transform_blocks,
        use_batch_norm=batch_norm,
        context_embedding_dim=context_embedding_dim
    )
