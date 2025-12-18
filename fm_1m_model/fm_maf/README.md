# Flow Matching with MAF (Masked Autoregressive Flow)

This directory contains the implementation of flow matching using Masked Autoregressive Flows (MAF) for gravitational wave parameter estimation.

## Overview

This method uses a conditional MAF model where a CNN processes the gravitational wave signal to create a context embedding, which conditions the autoregressive flow for parameter estimation.

## Files

- `train_maf_gaps.py`: Main training script
- `model_maf_gaps_new.py`: MAF model definition with CNN context processor
- `dataset_time.py`: Dataset loading and noise generation

## Usage

### Training

```bash
python train_maf_gaps.py
```

### Configuration

Key parameters in `train_maf_gaps.py`:

- `DATA_INDEX`: Dataset identifier (default: "ln_30_DAY")
- `RUN_INDEX`: Run identifier for outputs
- `ADD_GAPS`: Enable/disable data gaps (default: True)
- `GAP_LAMBDA_VALUE`: Gap distribution parameter (default: 0.75)
- `NUM_EPOCHS`: Number of training epochs (default: 250)
- `BATCH_SIZE`: Batch size (default: 128)
- `HIDDEN_DIMS_LIST`: Hidden dimensions for each MAF block (default: [256, 128, 64])
- `NUM_TRANSFORM_BLOCKS`: Number of residual blocks per MAF transform (default: 2)
- `CONTEXT_EMBEDDING_DIM`: Dimension of CNN context embedding (default: 256)

### Outputs

Training outputs are saved in `outputs/MAF_run_{RUN_INDEX}/`:

- `model_{RUN_INDEX}.pt`: Trained model weights
- `loss_history_{RUN_INDEX}.png`: Loss plot
- `corner_plot_{RUN_INDEX}.png`: Posterior distribution visualization
- `model_structure_{RUN_INDEX}.txt`: Model architecture details
- `params_stats_{RUN_INDEX}.npz`: Parameter normalization statistics

## Architecture

The model consists of:

1. **Context Processor**: CNN that processes the signal into a fixed-size embedding
2. **MAF Transforms**: Sequence of masked affine autoregressive transforms
3. **Base Distribution**: Standard normal distribution

The context embedding is computed once per signal and shared across all MAF transforms, making the model efficient.

## Features

- Efficient context processing (computed once per signal)
- Flexible architecture (configurable hidden dimensions and transform blocks)
- Handles data gaps in gravitational wave signals
- Noise curriculum learning
- GPU acceleration support
- Direct sampling from the learned distribution

