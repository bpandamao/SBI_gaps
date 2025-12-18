# Decoupled Flow Matching with DCAE

This directory contains a two-stage approach to flow matching: first training a Denoising Convolutional Autoencoder (DCAE) to compress signals, then training a flow matching model on the compressed representations.

## Overview

This method decouples signal processing from parameter estimation:

1. **Stage 1**: Train a DCAE to compress gravitational wave signals into a 256-dimensional bottleneck
2. **Stage 2**: Train a flow matching model conditioned on the bottleneck representations
3. **Stage 3**: Perform posterior analysis using the trained models

## Files

### Stage 1 (DCAE Training)
- `two_stage_main_stage_1_dcae.py`: Main script for Stage 1
- `two_stage_dcae.py`: DCAE model definition (encoder + decoder)
- `two_stage_dcae_trainer.py`: DCAE training loop

### Stage 2 (Flow Matching)
- `two_stage_main_stage_2_flow.py`: Main script for Stage 2
- `two_stage_flow_matcher_bottleneck.py`: Flow matching model for bottleneck conditioning
- `two_stage_flow_trainer_bottleneck.py`: Flow matching training loop

### Stage 3 (Posterior Analysis)
- `two_stage_main_stage_3_posterior.py`: Posterior analysis script
- `two_stage_posterior_bottleneck.py`: Posterior sampling and visualization

### Utilities
- `two_stage_dataset_time.py`: Dataset loading and noise generation
- `two_stage_trainer_gaps.py`: Gap creation and signal standardization utilities
- `residual_net.py`: Residual network implementation
- `model_utils_two_stage.py`: Model structure printing utilities

## Usage

### Stage 1: Train DCAE

```bash
python two_stage_main_stage_1_dcae.py
```

This trains the DCAE to compress signals. The trained encoder will be used in Stage 2.

**Configuration**:
- `DCAE_EPOCHS`: Number of epochs (default: 150)
- `DCAE_LR`: Learning rate (default: 1e-4)
- `PHYSICAL_BATCH_SIZE`: Batch size for GPU (default: 64)
- `LOGICAL_BATCH_SIZE`: Effective batch size with accumulation (default: 128)
- `BOTTLENECK_DIM`: Bottleneck dimension (default: 256)

**Outputs**: Saved in `outputs/run_{RUN_INDEX}/dcae/`

### Stage 2: Train Flow Matcher

```bash
python two_stage_main_stage_2_flow.py
```

This trains the flow matching model using the frozen encoder from Stage 1.

**Configuration**:
- `FM_EPOCHS`: Number of epochs (default: 300)
- `FM_LR`: Learning rate (default: 1e-4)
- `BATCH_SIZE`: Batch size (default: 128)

**Outputs**: Saved in `outputs/run_{RUN_INDEX}/flow_matcher2/`

### Stage 3: Posterior Analysis

```bash
python two_stage_main_stage_3_posterior.py
```

This performs posterior analysis on a test signal.

**Configuration**:
- `NUM_POSTERIOR_SAMPLES`: Number of samples (default: 3000)
- `CUSTOM_AMPLITUDE`, `CUSTOM_FREQUENCY`, `CUSTOM_FREQUENCY_DERIV`: Test signal parameters

**Outputs**: Saved in `outputs/run_{RUN_INDEX}/`

## Architecture

### DCAE (Stage 1)
- **Encoder**: 4-layer 1D CNN that compresses signals to 256 dimensions
- **Decoder**: 4-layer 1D transposed CNN that reconstructs signals
- **Loss**: MSE between original and reconstructed signals

### Flow Matcher (Stage 2)
- **Conditioning**: Bottleneck (256 dim) + log signal std (1 dim) = 257 dim
- **Flow Model**: Residual network that predicts flow velocity
- **Training**: Encoder is frozen, only flow model is trained

## Features

- **Memory Efficient**: Two-stage approach reduces memory requirements
- **Gap-Aware**: Handles data gaps in both stages
- **Gradient Accumulation**: Supports large effective batch sizes
- **Vectorized Operations**: Efficient batch processing for noise and gaps
- **Scale Information**: Includes signal scale (log std) in conditioning

## Advantages

1. **Reduced Memory**: Flow matching operates on 256-dim vectors instead of full signals
2. **Faster Training**: Smaller model in Stage 2 trains faster
3. **Better Generalization**: DCAE learns robust signal representations
4. **Modularity**: Can reuse encoder for other tasks

## Notes

- Stage 2 requires Stage 1 to complete first
- Stage 3 requires both Stage 1 and Stage 2 to complete
- The encoder is frozen during Stage 2 training
- All stages support data gaps and noise curriculum

