# Flow Matching for Gravitational Wave Parameter Estimation

This repository contains implementations of three different flow matching methods for gravitational wave parameter estimation from LISA data. All methods are designed to handle data gaps and noise in gravitational wave signals.

## Project Structure

```
fm_1m_model/
├── training_data_generator_30day_new.py  # Data generation script
├── fm_1m_conv1d/                         # Flow matching with Conv1D
├── fm_maf/                               # Flow matching with MAF (Masked Autoregressive Flow)
└── fm_decoupled_method/                  # Decoupled flow matching with DCAE
```

## Overview

This project implements three different approaches to flow matching for gravitational wave parameter estimation:

1. **Conv1D Flow Matching** (`fm_1m_conv1d/`): Uses 1D convolutional neural networks to process time-domain signals directly
2. **MAF Flow Matching** (`fm_maf/`): Uses Masked Autoregressive Flows for parameter estimation
3. **Decoupled Flow Matching** (`fm_decoupled_method/`): Two-stage approach using a Denoising Convolutional Autoencoder (DCAE) followed by flow matching

## Data Generation

First, generate the training data using:

```bash
python training_data_generator_30day_new.py
```

This will create a dataset of gravitational wave signals with corresponding parameters in the `training_data/` directory.

## Method-Specific Documentation

Each method has its own README with detailed instructions:

- [Conv1D Flow Matching](fm_1m_conv1d/README.md)
- [MAF Flow Matching](fm_maf/README.md)
- [Decoupled Flow Matching](fm_decoupled_method/README.md)

## Requirements

The project requires the following dependencies:

- PyTorch
- NumPy
- CuPy (for GPU acceleration)
- flow-matching
- glasflow (for MAF)
- lisatools
- fastlisaresponse
- matplotlib
- corner (for posterior visualization)
- tqdm

## Common Features

All methods support:

- **Data gaps**: Simulated gaps in gravitational wave data
- **Noise curriculum**: Progressive noise level during training
- **Posterior analysis**: Parameter estimation with uncertainty quantification
- **GPU acceleration**: CUDA support for faster training

## Citation

If you use this code, please cite the associated paper (if available).

## License

[Add your license here]

