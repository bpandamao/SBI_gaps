# Flow Matching with Conv1D

This directory contains the implementation of flow matching using 1D convolutional neural networks for gravitational wave parameter estimation.

## Overview

This method uses a 1D CNN to embed time-domain gravitational wave signals, which are then used to condition a flow matching model for parameter estimation. The architecture processes the full signal length (518,400 samples for 30 days at 5-second sampling) through convolutional layers to extract features.

## Files

- `main_gaps.py`: Main training script
- `flow_matcher_time.py`: Flow matching model definition
- `residual_net.py`: Residual network and signal embedding architectures
- `trainer_gaps.py`: Training loop with gap support
- `dataset_time.py`: Dataset loading and noise generation
- `posterior.py`: Posterior analysis and visualization
- `model_utils.py`: Model structure printing utilities

## Usage

### Training

```bash
python main_gaps.py
```

### Configuration

Key parameters in `main_gaps.py`:

- `DATA_INDEX`: Dataset identifier (default: "ln_30_DAY")
- `RUN_INDEX`: Run identifier for outputs
- `ADD_GAPS`: Enable/disable data gaps (default: True)
- `GAP_LAMBDA_VALUE`: Gap distribution parameter (default: 0.75)
- `NUM_EPOCHS`: Number of training epochs (default: 300)
- `BATCH_SIZE`: Batch size (default: 128)

### Outputs

Training outputs are saved in `outputs/run_{RUN_INDEX}/`:

- `flow_matcher_{RUN_INDEX}.pt`: Trained model weights
- `training_losses_{RUN_INDEX}.txt`: Training loss history
- `loss_history_{RUN_INDEX}.png`: Loss plot
- `posterior_{RUN_INDEX}.png`: Posterior distribution visualization
- `model_structure_{RUN_INDEX}.txt`: Model architecture details

## Architecture

The model consists of:

1. **Signal Embedding**: 1D CNN with 4 convolutional blocks that reduce the signal length
2. **Parameter Embedding**: Residual network processing time and parameter embeddings
3. **Flow Model**: Residual network that combines embeddings to predict flow velocity

## Features

- Handles data gaps in gravitational wave signals
- Noise curriculum learning
- GPU acceleration support
- Checkpoint saving and resuming
- Posterior analysis with corner plots

