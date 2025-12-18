# Flow Matching for Gravitational Wave Parameter Estimation

This repository contains implementations of flow matching methods for gravitational wave parameter estimation from LISA (Laser Interferometer Space Antenna) data. The codebase includes multiple approaches to handle data gaps, noise, and different signal representations for robust parameter inference.

## 🌟 Overview

This repository provides two main approaches to gravitational wave parameter estimation using flow matching:

1. **Time-Domain Flow Matching** (`fm_1m_model/`): Three different architectures operating directly on time-domain signals
2. **Wavelet Spectrogram Flow Matching** (`fm_3m_wavelet/`): Flow matching on wavelet-transformed spectrograms with advanced CNN architectures

Both approaches are designed to handle realistic LISA data conditions including:
- **Data gaps**: Simulated gaps in gravitational wave observations
- **Noise curriculum**: Progressive noise level during training
- **Posterior analysis**: Uncertainty quantification for parameter estimation
- **GPU acceleration**: CUDA support for efficient training

## 📁 Repository Structure

```
github/
├── fm_1m_model/              # Time-domain flow matching methods
│   ├── fm_1m_conv1d/         # Conv1D-based flow matching
│   ├── fm_maf/               # Masked Autoregressive Flow (MAF)
│   ├── fm_decoupled_method/  # Two-stage DCAE + flow matching
│   └── training_data_generator_30day_new.py
│
└── fm_3m_wavelet/            # Wavelet spectrogram flow matching
    ├── demo_complete_pipeline.py  # Complete end-to-end demo
    ├── training_data_generator_time_ln_likevb_smaller.py
    ├── augment_and_wavelet_gaps_ite_01_VB.py
    └── [model and training files]
```

## 🚀 Quick Start

### Time-Domain Methods (`fm_1m_model/`)

Generate training data first:
```bash
cd fm_1m_model
python training_data_generator_30day_new.py
```

Then train any of the three methods:
- **Conv1D**: `cd fm_1m_conv1d && python main_gaps.py`
- **MAF**: `cd fm_maf && python train_maf_gaps.py`
- **Decoupled**: Follow the three-stage process in `fm_decoupled_method/`

### Wavelet Spectrogram Method (`fm_3m_wavelet/`)

Run the complete pipeline:
```bash
cd fm_3m_wavelet
python demo_complete_pipeline.py --kernel_type asymmetric --num_samples 100 --num_epochs 50
```

## 📊 Methods Comparison

### Time-Domain Methods (`fm_1m_model/`)

| Method | Architecture | Key Features |
|--------|-------------|--------------|
| **Conv1D** | 1D CNN + Flow Matching | Direct time-domain processing, handles full signal length (518K samples) |
| **MAF** | CNN + Masked Autoregressive Flow | Efficient context processing, flexible architecture |
| **Decoupled** | DCAE (256-dim) + Flow Matching | Memory efficient, two-stage compression approach |

### Wavelet Spectrogram Method (`fm_3m_wavelet/`)

| Kernel Type | Architecture | Key Features |
|------------|-------------|--------------|
| **Symmetric** | 3×3 CNN kernels | Standard convolutions, faster training |
| **Asymmetric** | 3×9 CNN kernels + Dilation | Better temporal modeling, dilated convolutions [1,2,4,8] |

## 🔬 Key Features

### Data Handling
- **30-day observations**: Full LISA observation periods (518,400 samples at 5s sampling)
- **Gap simulation**: Realistic data gaps using exponential distributions
- **Noise injection**: LISA sensitivity curves with curriculum learning
- **Augmentation**: Multiple noise realizations per signal

### Model Architectures
- **Signal embedding**: CNN-based feature extraction from signals/spectrograms
- **Flow matching**: Continuous normalizing flows for parameter estimation
- **Posterior sampling**: Efficient sampling from learned distributions
- **Uncertainty quantification**: Full posterior distributions with corner plots

### Training Features
- **GPU acceleration**: CUDA support with automatic CPU fallback
- **Checkpointing**: Model saving and resuming
- **Loss visualization**: Training progress monitoring
- **Gradient accumulation**: Support for large effective batch sizes

## 📚 Documentation

Each subdirectory contains detailed README files:

- [`fm_1m_model/README.md`](fm_1m_model/README.md) - Overview of time-domain methods
- [`fm_1m_model/fm_1m_conv1d/README.md`](fm_1m_model/fm_1m_conv1d/README.md) - Conv1D method details
- [`fm_1m_model/fm_maf/README.md`](fm_1m_model/fm_maf/README.md) - MAF method details
- [`fm_1m_model/fm_decoupled_method/README.md`](fm_1m_model/fm_decoupled_method/README.md) - Decoupled method details
- [`fm_3m_wavelet/README.md`](fm_3m_wavelet/README.md) - Wavelet spectrogram method details
- [`fm_3m_wavelet/KERNEL_TYPES.md`](fm_3m_wavelet/KERNEL_TYPES.md) - Kernel architecture details

## 🛠️ Requirements

### Core Dependencies
- Python 3.7+
- PyTorch ≥1.9.0
- NumPy ≥1.21.0
- CuPy (for GPU acceleration)

### Domain-Specific
- `lisatools` - LISA sensitivity curves
- `fastlisaresponse` - LISA response calculations
- `flow-matching` - Flow matching ODE solvers
- `glasflow` - MAF implementation (for `fm_maf/`)
- `pywavelet` - Wavelet transforms (for `fm_3m_wavelet/`)

### Visualization
- matplotlib
- corner (for posterior plots)

See [`fm_1m_model/requirements.txt`](fm_1m_model/requirements.txt) for a complete list.

## 📈 Use Cases

This codebase is designed for:

1. **Parameter Estimation**: Infer gravitational wave source parameters (amplitude, frequency, frequency derivative) from noisy LISA data
2. **Gap Handling**: Robust inference with missing data segments
3. **Method Comparison**: Compare different flow matching architectures
4. **Research**: Extend and modify flow matching approaches for gravitational waves

## 🔍 Output Files

After training, models produce:

- **Model weights**: `.pt` files with trained parameters
- **Loss history**: Training curves and statistics
- **Posterior samples**: Parameter distributions
- **Corner plots**: Visualization of parameter posteriors
- **Model structure**: Architecture details and parameter counts

## 📝 Citation

If you use this code, please cite the associated paper (if available) and acknowledge the use of:
- Flow matching methods
- LISA tools (`lisatools`, `fastlisaresponse`)
- PyWavelet (for wavelet transforms)

## 🤝 Contributing

This is a research codebase. For questions or issues, please open an issue on GitHub.

## 📄 License

[Add your license here]

---

## 🎯 Method Selection Guide

**Choose `fm_1m_model/` if:**
- You want to work directly with time-domain signals
- You need memory-efficient training (decoupled method)
- You prefer simpler architectures (Conv1D)

**Choose `fm_3m_wavelet/` if:**
- You want to leverage frequency-domain information
- You need better temporal modeling (asymmetric kernels)
- You want a complete, easy-to-use pipeline

