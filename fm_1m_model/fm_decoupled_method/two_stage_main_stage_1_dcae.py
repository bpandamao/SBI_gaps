import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support

"""
Stage 1: Train Denoising Convolutional Autoencoder (DCAE).

This script trains a DCAE to compress gravitational wave signals into a
bottleneck representation, which will be used in Stage 2 for flow matching.
"""

from two_stage_dataset_time import prepare_data
from two_stage_dcae import DCAE
from two_stage_dcae_trainer import train_dcae
from model_utils_two_stage import print_dcae_structure

# --- Configuration ---
RUN_INDEX = "TWO_STAGE_DCAE_FM"
DATA_INDEX = "ln_30_DAY" # Same data as your first project

# Stage 1 (DCAE) Config
DCAE_EPOCHS = 150
DCAE_LR = 1e-4
DCAE_CHECKPOINT_DIR = f"outputs/run_{RUN_INDEX}/dcae"
DCAE_MODEL_FILE = os.path.join(DCAE_CHECKPOINT_DIR, "dcae_final.pt")
DCAE_LOSS_PLOT_FILE = os.path.join(DCAE_CHECKPOINT_DIR, f"dcae_loss_{RUN_INDEX}.png")
DCAE_STRUCTURE_FILE = os.path.join(DCAE_CHECKPOINT_DIR, f"dcae_structure_{RUN_INDEX}.txt") # <-- ADDED
RESUME_DCAE = False
START_EPOCH_DCAE = 0
NORMALIZATION_FILE = os.path.join(f"outputs/run_{RUN_INDEX}", "model_info.npz")

# Memory Configuration
# Set physical batch size to fit on GPU
PHYSICAL_BATCH_SIZE = 64
# Set logical batch size for stable training
LOGICAL_BATCH_SIZE = 128
# Calculate accumulation steps
ACCUMULATION_STEPS = LOGICAL_BATCH_SIZE // PHYSICAL_BATCH_SIZE

# General Config
SIGNAL_LENGTH = 518400 # 30 days * 24 * 60 * 60 / 5s
DT = 5
BOTTLENECK_DIM = 256
ADD_GAPS = True
GAP_LAMBDA_VALUE = 0.75
# --- End Configuration ---

def plot_losses(train_losses, test_losses, save_path, title):
    """Helper function to plot and save loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def standardize_signal(signal):
    """
    Standardize a signal by subtracting mean and dividing by std.
    Add a small epsilon to avoid division by zero for silent signals (gaps).
    """
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    return (signal - mean) / (std)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(DCAE_CHECKPOINT_DIR, exist_ok=True)
    
    # --- Load Data ---
    try:
        data = np.load(f'training_data/training_set_{DATA_INDEX}.npz')
        parameters = data['parameters']
        time_signals = data['time_signals']
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    time_signals_reshaped = time_signals.reshape(time_signals.shape[0], 1, time_signals.shape[1])
    
    # Note: We create data loaders *without* device specified,
    # as the trainers will move data. This is better for multiprocessing.
    train_loader, test_loader, params_mean, params_std = prepare_data(
        parameters, 
        time_signals_reshaped,
        batch_size=PHYSICAL_BATCH_SIZE # <-- CHANGED
    )
    
    # Save normalization info
    np.savez(NORMALIZATION_FILE, 
             params_mean=params_mean, params_std=params_std)
    print(f"Saved normalization info to {NORMALIZATION_FILE}")

    # ========================================
    # --- STAGE 1: Train DCAE ---
    # ========================================
    print("\n" + "="*30)
    print("--- STARTING STAGE 1: DCAE Training ---")
    print("="*30)
    
    dcae_model = DCAE(
        in_channels=1, 
        bottleneck_dim=BOTTLENECK_DIM, 
        signal_length=SIGNAL_LENGTH
    ).to(device)
    
    # Print model structure
    print(f"Physical Batch Size: {PHYSICAL_BATCH_SIZE}")
    print(f"Logical Batch Size: {LOGICAL_BATCH_SIZE}")
    print(f"Accumulation Steps: {ACCUMULATION_STEPS}")
    print_dcae_structure(dcae_model, filename=DCAE_STRUCTURE_FILE)
    
    start_dcae_training = time.time()
    
    if RESUME_DCAE:
        checkpoint_path = os.path.join(DCAE_CHECKPOINT_DIR, f'dcae_checkpoint_epoch_{START_EPOCH_DCAE}.pt')
        if os.path.exists(checkpoint_path):
             print(f"Attempting to resume DCAE from {checkpoint_path}")
        else:
             print(f"Resume requested, but checkpoint not found. Starting DCAE from scratch.")
             START_EPOCH_DCAE = 0
    
    dcae_model, dcae_train_losses, dcae_test_losses = train_dcae(
        dcae_model,
        train_loader,
        test_loader,
        device,
        num_epochs=DCAE_EPOCHS,
        lr=DCAE_LR,
        checkpoint_dir=DCAE_CHECKPOINT_DIR,
        start_epoch=START_EPOCH_DCAE if RESUME_DCAE else 0,
        signal_length=SIGNAL_LENGTH,
        dt=DT,
        add_gaps=ADD_GAPS,
        gap_lambda_value=GAP_LAMBDA_VALUE,
        accumulation_steps=ACCUMULATION_STEPS
    )
    
    end_dcae_training = time.time()
    print(f"--- STAGE 1 (DCAE) Complete ---")
    print(f"Training time: {(end_dcae_training - start_dcae_training):.2f} seconds")
    
    # Save final DCAE model
    torch.save(dcae_model.state_dict(), DCAE_MODEL_FILE)
    print(f"Saved final DCAE model to {DCAE_MODEL_FILE}")
    
    # Plot DCAE losses
    plot_losses(dcae_train_losses, dcae_test_losses, DCAE_LOSS_PLOT_FILE, "DCAE Training and Validation Loss")
    print(f"Saved DCAE loss plot to {DCAE_LOSS_PLOT_FILE}")
    print(f"\nStage 1 finished. You can now run main_stage_2_flow.py")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
    
    try:
        if __name__ == "__main__":
            freeze_support()
    except Exception:
        pass