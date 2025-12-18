# Kernel Types: Symmetric vs Asymmetric

This document explains the differences between symmetric and asymmetric kernel implementations in the flow matching models.

## Overview

The codebase provides two implementations of convolutional kernels for processing wavelet spectrograms:

1. **Symmetric Kernels** (`residual_wavelet_try01.py` + `flow_matcher_time.py`)
2. **Asymmetric Kernels** (`residual_wavelet_try_asycnn_dilated3339.py` + `flow_matcher_time_asy.py`)

## Symmetric Kernels

### Architecture
- **File**: `residual_wavelet_try01.py`
- **Kernel Size**: 3×3 (square, symmetric)
- **Dilation**: None (standard convolutions)
- **Flow Matcher**: `flow_matcher_time.py`

### Characteristics
- Equal receptive field in both frequency and time dimensions
- Standard convolution operations
- Simpler architecture
- Good for isotropic features

### Code Location
```python
# In Conv2dResidualBlock
self.main_path = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, ...),
    ...
)
```

### Layer Structure
```
Initial Layer: 3×3 conv, stride=2
Layer 1: 64 channels, 2 blocks, stride=1
Layer 2: 128 channels, 2 blocks, stride=2
Layer 3: 256 channels, 1 block, stride=2
Layer 4: 512 channels, 1 block, stride=2
```

## Asymmetric Kernels

### Architecture
- **File**: `residual_wavelet_try_asycnn_dilated3339.py`
- **Kernel Size**: 3×9 (asymmetric: 3 freq × 9 time)
- **Dilation**: Progressive [1, 2, 4, 8]
- **Flow Matcher**: `flow_matcher_time_asy.py`

### Characteristics
- Wider receptive field in time dimension (9 vs 3)
- Dilated convolutions for long-range dependencies
- Better temporal modeling
- More parameters

### Code Location
```python
# In Conv2dResidualBlock
kernel_size = (3, 9)  # Asymmetric: 3 freq × 9 time
dilation = (1, dilation_rate)  # Dilation only on time axis
padding = ((kernel_size[0] - 1) // 2, dilation_rate * (kernel_size[1] - 1) // 2)

self.main_path = nn.Sequential(
    nn.Conv2d(..., kernel_size=kernel_size, dilation=dilation, padding=padding, ...),
    ...
)
```

### Layer Structure with Dilation
```
Initial Layer: 3×3 conv, stride=2
Layer 1: 64 channels, 2 blocks, stride=1, dilation=1
Layer 2: 128 channels, 2 blocks, stride=2, dilation=2
Layer 3: 256 channels, 1 block, stride=2, dilation=4
Layer 4: 512 channels, 1 block, stride=2, dilation=8
```

## Key Differences

| Feature | Symmetric | Asymmetric |
|---------|-----------|------------|
| Kernel Size | 3×3 | 3×9 |
| Dilation | None | Progressive [1,2,4,8] |
| Receptive Field | Equal in both dims | Wider in time |
| Parameters | Fewer | More |
| Temporal Modeling | Standard | Enhanced |
| Use Case | Isotropic features | Temporal patterns |

## Dilation Strategy

The asymmetric model uses **progressive dilation** to increase the receptive field:

- **Layer 1** (dilation=1): Captures local patterns
- **Layer 2** (dilation=2): Medium-range dependencies
- **Layer 3** (dilation=4): Longer-range patterns
- **Layer 4** (dilation=8): Very long-range dependencies

This allows the model to capture both:
- **Local features**: Fine-grained patterns in the spectrogram
- **Global features**: Long-term temporal trends

## When to Use Which?

### Use Symmetric Kernels When:
- Features are isotropic (similar in frequency and time)
- Computational resources are limited
- Simpler model is preferred
- Data has balanced frequency/time structure

### Use Asymmetric Kernels When:
- Temporal patterns are important
- Long-range dependencies matter
- You have sufficient computational resources
- Data has strong temporal structure (e.g., gravitational wave signals)

## Implementation Details

### Padding Calculation

For asymmetric kernels with dilation, padding is calculated as:
```python
padding_freq = (kernel_size[0] - 1) // 2  # Standard padding
padding_time = dilation_rate * (kernel_size[1] - 1) // 2  # Dilated padding
```

This ensures the output size matches the input size (when stride=1).

### Receptive Field

The effective receptive field grows with dilation:
- Layer 1: ~9 time steps
- Layer 2: ~18 time steps  
- Layer 3: ~36 time steps
- Layer 4: ~72 time steps

Combined with striding, the final receptive field can cover a significant portion of the input spectrogram.

## Performance Considerations

- **Memory**: Asymmetric kernels use more memory due to larger kernels
- **Computation**: Asymmetric kernels are slower but provide better temporal modeling
- **Convergence**: Asymmetric kernels may converge faster for temporal tasks
- **Generalization**: Asymmetric kernels may generalize better to long signals

## Example Usage

```python
# Symmetric
from flow_matcher_time import ContinuousFlowMatcherTime as SymmetricModel
model = SymmetricModel(param_dim=3, signal_embedding_dim=512)

# Asymmetric  
from flow_matcher_time_asy import ContinuousFlowMatcherTime as AsymmetricModel
model = AsymmetricModel(param_dim=3, signal_embedding_dim=512)
```

Both models have the same interface, so switching is straightforward!

