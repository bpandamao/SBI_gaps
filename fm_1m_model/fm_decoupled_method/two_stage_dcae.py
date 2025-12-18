import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    1D CNN Encoder to compress the signal to a 256-dim bottleneck.
    Based on the structure in your residual_net.py but simplified for this task.
    """
    def __init__(self, in_channels=1, bottleneck_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Input: [B, 1, 518400]
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=4, padding=7), # [B, 64, 129600]
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=15, stride=6, padding=7), # [B, 128, 21600]
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=17, stride=8, padding=8), # [B, 256, 2700]
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=21, stride=10, padding=10), # [B, 512, 270]
            nn.BatchNorm1d(512),
            nn.GELU()
        )
        
        # Adaptive pooling to handle any minor length variations
        self.final_pool = nn.AdaptiveAvgPool1d(1) # [B, 512, 1]
        
        # Projection to the final bottleneck dimension
        self.final_proj = nn.Linear(512, bottleneck_dim) # [B, 256]

    def forward(self, x):
        # x shape: [B, 1, SignalLength]
        x = self.conv_layers(x)
        x = self.final_pool(x) # [B, 512, 1]
        x = x.view(x.size(0), -1) # Flatten: [B, 512]
        x = self.final_proj(x) # [B, 256]
        return x

class Decoder(nn.Module):
    """
    1D Transposed CNN Decoder to reconstruct the signal from the 256-dim bottleneck.
    This architecture mirrors the Encoder.
    """
    def __init__(self, bottleneck_dim=256, out_channels=1, signal_length=518400):
        super().__init__()
        self.signal_length = signal_length
        
        # Initial projection from bottleneck to a shape suitable for transposed conv
        self.initial_proj = nn.Linear(bottleneck_dim, 512 * 270) # Match Encoder's [B, 512, 270]
        
        self.trans_conv_layers = nn.Sequential(
            # Input: [B, 512, 270]
            # Block 1
            nn.ConvTranspose1d(512, 256, kernel_size=21, stride=10, padding=10, output_padding=9), # [B, 256, 2700] (check output_padding)
            nn.BatchNorm1d(256),
            nn.GELU(),

            # Block 2
            nn.ConvTranspose1d(256, 128, kernel_size=17, stride=8, padding=8, output_padding=7), # [B, 128, 21600]
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            # Block 3
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=6, padding=7, output_padding=5), # [B, 64, 129600]
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Block 4
            nn.ConvTranspose1d(64, out_channels, kernel_size=15, stride=4, padding=7, output_padding=3), # [B, 1, 518400]
        )
        
        # Final layer to ensure exact signal length
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # x shape: [B, 256]
        x = self.initial_proj(x) # [B, 512 * 270]
        x = x.view(x.size(0), 512, 270) # [B, 512, 270]
        
        x = self.trans_conv_layers(x) # [B, 1, 518400]
        
        # Ensure exact output length
        current_len = x.shape[2]
        if current_len > self.signal_length:
            x = x[:, :, :self.signal_length]
        elif current_len < self.signal_length:
            padding = self.signal_length - current_len
            x = nn.functional.pad(x, (0, padding))

        x = self.final_conv(x) # Final touch, [B, 1, 518400]
        return x

class DCAE(nn.Module):
    """
    Denoising Convolutional Autoencoder.
    """
    def __init__(self, in_channels=1, bottleneck_dim=256, signal_length=518400):
        super().__init__()
        self.encoder = Encoder(in_channels, bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim, in_channels, signal_length)

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed_signal = self.decoder(bottleneck)
        return reconstructed_signal, bottleneck