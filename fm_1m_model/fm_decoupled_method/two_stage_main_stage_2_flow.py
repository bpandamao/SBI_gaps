import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support

"""
Stage 2: Train Flow Matcher on Bottleneck Representations.

This script trains a flow matching model conditioned on the bottleneck
representations from the DCAE encoder trained in Stage 1.
"""

from two_stage_dataset_time import prepare_data
from two_stage_dcae import DCAE
from two_stage_flow_matcher_bottleneck import ContinuousFlowMatcherBottleneck
from two_stage_flow_trainer_bottleneck import train_flow_bottleneck
from model_utils_two_stage import print_flow_bottleneck_structure

# --- Configuration ---
RUN_INDEX = "TWO_STAGE_DCAE_FM"
DATA_INDEX = "ln_30_DAY" # Same data as your first project

# Stage 1 (DCAE) Config - Needed for loading
DCAE_CHECKPOINT_DIR = f"outputs/run_{RUN_INDEX}/dcae"
DCAE_MODEL_FILE = os.path.join(DCAE_CHECKPOINT_DIR, "dcae_final.pt")
NORMALIZATION_FILE = os.path.join(f"outputs/run_{RUN_INDEX}", "model_info.npz")

# Stage 2 (Flow Matcher) Config
FM_EPOCHS = 300
FM_LR = 1e-4
FM_CHECKPOINT_DIR = f"outputs/run_{RUN_INDEX}/flow_matcher"
FM_MODEL_FILE = os.path.join(FM_CHECKPOINT_DIR, "flow_matcher_bottleneck_final.pt")
FM_LOSS_PLOT_FILE = os.path.join(FM_CHECKPOINT_DIR, f"fm_loss_{RUN_INDEX}.png")
FM_STRUCTURE_FILE = os.path.join(FM_CHECKPOINT_DIR, f"fm_structure_{RUN_INDEX}.txt")
RESUME_FM = False
START_EPOCH_FM = 0

# General Config
BATCH_SIZE = 128 
SIGNAL_LENGTH = 518400 # 30 days * 24 * 60 * 60 / 5s
DT = 5
BOTTLENECK_DIM = 256
ADD_GAPS = True
GAP_LAMBDA_VALUE = 0.75

# --- All Stage 3 (Posterior Analysis) constants have been removed ---
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(FM_CHECKPOINT_DIR, exist_ok=True)
    
    # --- Load Normalization Stats from Stage 1 ---
    try:
        norm_data = np.load(NORMALIZATION_FILE)
        # We don't need these for training, but good to check file exists
        # params_mean = norm_data['params_mean']
        # params_std = norm_data['params_std']
        print(f"Located normalization info at {NORMALIZATION_FILE}")
    except Exception as e:
        print(f"Error loading normalization file '{NORMALIZATION_FILE}': {e}")
        print("Please run main_stage_1_dcae.py first!")
        return
        
    # --- Load Data (needed for loaders) ---
    try:
        data = np.load(f'training_data/training_set_{DATA_INDEX}.npz')
        parameters = data['parameters']
        time_signals = data['time_signals']
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    time_signals_reshaped = time_signals.reshape(time_signals.shape[0], 1, time_signals.shape[1])
    
    train_loader, test_loader, _, _ = prepare_data(
        parameters, 
        time_signals_reshaped,
        batch_size=BATCH_SIZE
    )
    print("Data loaders prepared.")

    # --- Load Pre-trained Encoder from Stage 1 ---
    print(f"Loading pre-trained DCAE model from {DCAE_MODEL_FILE}...")
    try:
        # Instantiate the full DCAE model first
        dcae_model = DCAE(
            in_channels=1, 
            bottleneck_dim=BOTTLENECK_DIM, 
            signal_length=SIGNAL_LENGTH
        ).to(device)
        
        # Load the state dict
        dcae_model.load_state_dict(torch.load(DCAE_MODEL_FILE, map_location=device))
        
        # Extract the encoder
        trained_encoder = dcae_model.encoder
        trained_encoder.eval() # Set to evaluation mode
        print("Successfully loaded trained encoder from Stage 1.")
    except Exception as e:
        print(f"Error loading DCAE model file: {e}")
        print("Please ensure main_stage_1_dcae.py has run successfully.")
        return

    # ========================================
    # --- STAGE 2: Train Flow Matcher ---
    # ========================================
    print("\n" + "="*30)
    print("--- STARTING STAGE 2: Flow Matcher Training ---")
    print("="*30)
    
    # We only need the encoder part for Stage 2
    trained_encoder = dcae_model.encoder
    
    flow_model = ContinuousFlowMatcherBottleneck(
        param_dim=parameters.shape[1],
        conditioning_dim=BOTTLENECK_DIM + 1 # 256 + 1
    ).to(device)
    
    # --- Print model structure ---
    print_flow_bottleneck_structure(flow_model, filename=FM_STRUCTURE_FILE)
    
    start_fm_training = time.time()
    
    if RESUME_FM:
        checkpoint_path = os.path.join(FM_CHECKPOINT_DIR, f'flow_checkpoint_epoch_{START_EPOCH_FM}.pt')
        if os.path.exists(checkpoint_path):
             print(f"Attempting to resume Flow Matcher from {checkpoint_path}")
        else:
             print(f"Resume requested, but checkpoint not found. Starting Flow Matcher from scratch.")
             START_EPOCH_FM = 0

    flow_model, fm_train_losses, fm_test_losses = train_flow_bottleneck(
        flow_model,
        trained_encoder, # Pass the frozen, pre-trained encoder
        train_loader,
        test_loader,
        device,
        num_epochs=FM_EPOCHS,
        lr=FM_LR,
        checkpoint_dir=FM_CHECKPOINT_DIR, # Use the dedicated FM checkpoint dir
        start_epoch=START_EPOCH_FM if RESUME_FM else 0,
        signal_length=SIGNAL_LENGTH,
        dt=DT,
        add_gaps=ADD_GAPS,
        gap_lambda_value=GAP_LAMBDA_VALUE
    )
    
    end_fm_training = time.time()
    print(f"--- STAGE 2 (Flow Matcher) Complete ---")
    print(f"Training time: {(end_fm_training - start_fm_training):.2f} seconds")

    # Save final Flow Matcher model
    torch.save(flow_model.state_dict(), FM_MODEL_FILE)
    print(f"Saved final Flow Matcher model to {FM_MODEL_FILE}")
    
    # Plot Flow Matcher losses
    plot_losses(fm_train_losses, fm_test_losses, FM_LOSS_PLOT_FILE, "Flow Matcher (Bottleneck) Training and Validation Loss")
    print(f"Saved Flow Matcher loss plot to {FM_LOSS_PLOT_FILE}")

    print("\nStage 2 training completed successfully!")
    print(f"To run posterior analysis, run: python two_stage_main_stage_3_posterior.py")


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