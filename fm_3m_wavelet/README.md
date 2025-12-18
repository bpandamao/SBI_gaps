# Flow Matching with Wavelet Spectrograms

This directory contains code for training flow matching models on wavelet-transformed gravitational wave signals with data gaps.

## Overview

The pipeline consists of three main steps:

1. **Training Data Generation**: Generate clean gravitational wave signals
2. **Data Augmentation**: Add noise and gaps to create realistic observations
3. **Flow Matching Training**: Train a neural network to learn the parameter posterior distribution

## Directory Structure

```
fm_3m_wavelet/
├── demo_complete_pipeline.py          # Complete demo script (recommended)
├── README.md                          # This file
├── DEPRECATED.md                      # Deprecated files guide (for reference)
├── CLEANUP_SUMMARY.md                 # Code cleanup documentation
│
├── Data Generation:
│   ├── training_data_generator_time_ln_likevb.py  # Generate clean signals
│   └── augment_and_wavelet_gaps_ite_01_VB.py              # Add noise & gaps
│
├── Dataset:
│   └── dataset_wavelet_sub_ite_01_update.py               # PyTorch dataset
│
├── Models:
│   ├── flow_matcher_time.py                               # Symmetric kernel flow matcher
│   ├── flow_matcher_time_asy.py                          # Asymmetric kernel flow matcher
│   ├── residual_wavelet_try01.py                         # Symmetric CNN backbone
│   └── residual_wavelet_try_asycnn_dilated3339.py        # Asymmetric CNN backbone (with dilation)
│
├── Training:
│   ├── trainer_spectrogramV1.py                          # Training loop
│   └── model_utils.py                                    # Model utilities
│
└── Inference:
    └── posterior01.py                                    # Posterior sampling
```

## Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```bash
# Symmetric kernels (3x3 convolutions with dilation)
python demo_complete_pipeline.py --kernel_type symmetric --num_samples 100 --num_epochs 50

# Asymmetric kernels (3x9 convolutions with dilation)
python demo_complete_pipeline.py --kernel_type asymmetric --num_samples 100 --num_epochs 50
```

### Advanced Usage

```bash
# Custom configuration
python demo_complete_pipeline.py \
    --kernel_type asymmetric \
    --num_samples 1000 \
    --num_augmentations 10 \
    --batch_size 64 \
    --num_epochs 200 \
    --signal_embedding_dim 512 \
    --output_dir my_experiment \
    --train_subset_ratio 0.5

# Use existing data (skip generation)
python demo_complete_pipeline.py \
    --kernel_type symmetric \
    --use_existing_data \
    --num_epochs 100
```

## Key Features

### Kernel Types

1. **Symmetric Kernels** (`residual_wavelet_try01.py`):
   - Uses 3x3 convolutional kernels
   - Symmetric receptive field
   - Suitable for isotropic features

2. **Asymmetric Kernels** (`residual_wavelet_try_asycnn_dilated3339.py`):
   - Uses 3x9 convolutional kernels
   - Asymmetric receptive field (wider in time dimension)
   - Better for temporal patterns
   - Includes dilation rates: [1, 2, 4, 8] for increasing receptive field

### Dilation Strategy

The asymmetric model uses progressively increasing dilation rates:
- Layer 1: dilation_rate = 1
- Layer 2: dilation_rate = 2
- Layer 3: dilation_rate = 4
- Layer 4: dilation_rate = 8

This allows the model to capture both local and long-range temporal dependencies.

## Step-by-Step Usage

### 1. Generate Training Data

```python
from training_data_generator_time_ln_likevb import generate_training_set

# Modify constants in the file or call directly
generate_training_set()
```

This creates a `.npz` file with:
- `parameters`: Parameter arrays (log amplitude, log frequency, log frequency derivative)
- `time_signals`: Clean time-domain signals
- `time_array`: Time array

### 2. Augment Data

```python
from augment_and_wavelet_gaps_ite_01_VB import augment_and_transform_parallel

# Modify constants in the file:
# - INPUT_DATA_FILE: path to .npz file from step 1
# - OUTPUT_H5_FILE: desired output path
# - NOISE_AUGMENTATIONS: number of augmentations per signal

augment_and_transform_parallel()
```

This creates an `.h5` file with:
- `parameters`: Original parameters
- `spectrograms`: Augmented spectrograms (with noise and gaps)
- Attributes: `spectrogram_min`, `spectrogram_max` for normalization

### 3. Train Flow Matching Model

```python
from flow_matcher_time_asy import ContinuousFlowMatcherTime
from dataset_wavelet_sub_ite_01_update import prepare_data_from_h5
from trainer_spectrogramV1 import train_flow_matching_spectrogram

# Load data
train_loader, test_loader = prepare_data_from_h5(
    train_h5_path="path/to/train.h5",
    test_h5_path="path/to/test.h5",
    parameters_min=params_min,
    parameters_max=params_max,
    batch_size=32
)

# Create model
model = ContinuousFlowMatcherTime(
    param_dim=3,
    signal_embedding_dim=512,
    signal_input_dim=1572864
).to(device)

# Train
model, train_losses, test_losses = train_flow_matching_spectrogram(
    model, train_loader, test_loader, device,
    num_epochs=100
)
```

## Model Architecture

### Flow Matching Model

The model consists of three main components:

1. **Signal Embedding Network**: 
   - Convolutional encoder (ResNet-style)
   - Extracts features from wavelet spectrograms
   - Output: `signal_embedding_dim` dimensional vector

2. **Parameter Embedding Network**:
   - MLP that embeds time `t` and parameters `θ`
   - Output: `param_embedding_dim` dimensional vector

3. **Flow Network**:
   - Combines signal and parameter embeddings
   - Predicts the velocity field for the flow matching ODE

### CNN Backbone Differences

**Symmetric** (`residual_wavelet_try01.py`):
- Kernel size: 3x3
- Standard convolutions

**Asymmetric** (`residual_wavelet_try_asycnn_dilated3339.py`):
- Kernel size: 3x9 (asymmetric)
- Dilated convolutions with increasing rates
- Better temporal modeling

## Output Files

After running the demo, you'll find:

```
demo_outputs/
├── training_data/
│   └── fullsignal_*.npz              # Generated clean signals
├── augmented_data/
│   └── *_augmented.h5                # Augmented spectrograms
└── model_{kernel_type}/
    ├── model_structure.txt           # Model architecture
    ├── model_info.npz                # Normalization parameters
    ├── flow_matcher.pt               # Trained model weights
    ├── losses.npz                    # Training history
    └── loss_plot.png                 # Loss visualization
```

## Dependencies

Required packages:
- `torch` (PyTorch)
- `numpy`
- `h5py`
- `matplotlib`
- `tqdm`
- `flow_matching` (for ODE solver)
- `lisatools` (for LISA sensitivity)
- `fastlisaresponse` (for LISA response)
- `pywavelet` (for wavelet transforms)

## Notes

- The code assumes GPU availability but will fall back to CPU
- Data generation and augmentation can be time-consuming for large datasets
- Use `--train_subset_ratio` to train on a subset for faster testing
- The model uses a two-stage learning rate schedule (70% epochs at 1e-3, 30% at 1e-4)
- See `DEPRECATED.md` for information about deprecated files
- See `CLEANUP_SUMMARY.md` for details on recent code cleanup

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed and paths are correct
2. **Memory issues**: Reduce `batch_size` or `num_samples`
3. **Slow training**: Use `--train_subset_ratio 0.1` for faster iteration
4. **File not found**: Check that data generation completed successfully

## Citation

If you use this code, please cite the relevant papers for:
- Flow matching
- LISA tools
- Wavelet transforms

