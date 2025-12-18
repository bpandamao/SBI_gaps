import torch
import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
import time
import os
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support

"""
Stage 3: Posterior Analysis.

This script performs posterior analysis using the trained DCAE encoder and
flow matcher from Stages 1 and 2.
"""

from two_stage_posterior_bottleneck import analyze_posterior
from two_stage_dcae import DCAE
from two_stage_flow_matcher_bottleneck import ContinuousFlowMatcherBottleneck
from lisatools.sensitivity import get_sensitivity, A1TDISens
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

# --- Configuration (Copied from main_stage_2_flow.py) ---
RUN_INDEX = "TWO_STAGE_DCAE_FM"
DATA_INDEX = "ln_30_DAY" # Same data as your first project

# Stage 1 (DCAE) Config - Needed for loading
DCAE_CHECKPOINT_DIR = f"outputs/run_{RUN_INDEX}/dcae"
DCAE_MODEL_FILE = os.path.join(DCAE_CHECKPOINT_DIR, "dcae_final.pt")
NORMALIZATION_FILE = os.path.join(f"outputs/run_{RUN_INDEX}", "model_info.npz")

# Stage 2 (Flow Matcher) Config
FM_CHECKPOINT_DIR = f"outputs/run_{RUN_INDEX}/flow_matcher"
FM_MODEL_FILE = os.path.join(FM_CHECKPOINT_DIR, "flow_matcher_bottleneck_final.pt")

# General Config
NUM_POSTERIOR_SAMPLES = 3000
SIGNAL_LENGTH = 518400 # 30 days * 24 * 60 * 60 / 5s
DT = 5
BOTTLENECK_DIM = 256
ADD_GAPS = True
GAP_LAMBDA_VALUE = 0.75

# Constants from your files
YRSID_SI = 31558149.763545603
IOTA = 1.11820901
PHI0 = 4.91128699
PSI = 2.3290324
BETA = 0.9805742971871619
LAM = 5.22979888
CUSTOM_AMPLITUDE = 4.8e-21
CUSTOM_FREQUENCY = 2e-3
CUSTOM_FREQUENCY_DERIV = 3.00002e-10
LN_CUSTOM_AMPLITUDE = np.log(CUSTOM_AMPLITUDE)
LN_CUSTOM_FREQUENCY = np.log(CUSTOM_FREQUENCY)
LN_CUSTOM_FREQUENCY_DERIV = np.log(CUSTOM_FREQUENCY_DERIV)
# --- End Configuration ---


# Helper Functions

class GBWave:
    """Gravitational wave signal generator."""
    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=5):
         # get the t array 
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc

def standardize_signal(signal):
    """Standardize signal by subtracting mean and dividing by std."""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero for silent signals (gaps)
    return (signal - mean) / (std)

def create_gaps_mask(duration, dt, lambda_value=0.75):
    """Create gaps mask for data."""
    total_points = int(duration / dt)
    mask = np.ones(total_points, dtype=np.float32)
    
    current_point = 0
    while current_point < total_points:
        # Duration of data segment follows an exponential distribution
        data_length_days = np.random.exponential(scale=1/lambda_value)
        data_length_points = int(data_length_days * 24 * 3600 / dt)

        current_point += data_length_points
        if current_point >= total_points:
            break

        # Duration of gap is uniform between 3.5 and 12 hours
        gap_duration_hours = np.random.uniform(3.5, 12)
        gap_duration_points = int(gap_duration_hours * 3600 / dt)

        gap_end_point = min(current_point + gap_duration_points, total_points)
        mask[current_point:gap_end_point] = 0
        
        current_point += gap_duration_points

    return mask

def generate_noise(dt, psd_tensor, noise_level=1, device=None, N=518400, batch_size=1):
    """Generate noise in time domain based on given PSD."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_t = int(2**(torch.ceil(torch.log2(torch.tensor(N, device=device)))))
    freqs = torch.fft.rfftfreq(N, dt, device=device)[1:]  # Exclude DC component
 
    # Calculate variance in frequency domain
    variance_noise_f = (N_t * psd_tensor / (4 * dt))
    
    # Generate complex Gaussian noise
    noise_f_real = torch.randn(batch_size, len(freqs), device=device)
    noise_f_imag = torch.randn(batch_size, len(freqs), device=device)
    noise_f = torch.sqrt(variance_noise_f) * (noise_f_real + 1j * noise_f_imag)
    
    # Add DC component (zero)
    noise_f = torch.cat((torch.zeros(batch_size, 1, device=device), noise_f), dim=1)
    
    # Transform to time domain
    noise_t = torch.fft.irfft(noise_f, n=N_t, dim=1)[:, :N] # [batch_size, N]
    
    output_noise_t = noise_t * noise_level
    # Return shape [batch_size, N]
    return output_noise_t


# --- Main Analysis Function ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load Normalization Stats ---
    try:
        norm_data = np.load(NORMALIZATION_FILE)
        params_mean = norm_data['params_mean']
        params_std = norm_data['params_std']
        print(f"Loaded normalization info from {NORMALIZATION_FILE}")
    except Exception as e:
        print(f"Error loading normalization file '{NORMALIZATION_FILE}': {e}")
        print("Please ensure main_stage_1_dcae.py has run successfully.")
        return
        
    # --- Load Pre-trained Encoder ---
    print(f"Loading pre-trained DCAE model from {DCAE_MODEL_FILE}...")
    try:
        dcae_model = DCAE(
            in_channels=1, 
            bottleneck_dim=BOTTLENECK_DIM, 
            signal_length=SIGNAL_LENGTH
        ).to(device)
        dcae_model.load_state_dict(torch.load(DCAE_MODEL_FILE, map_location=device))
        trained_encoder = dcae_model.encoder
        trained_encoder.eval() 
        print("Successfully loaded trained encoder from Stage 1.")
    except Exception as e:
        print(f"Error loading DCAE model file: {e}")
        return

    # --- Load Pre-trained Flow Matcher ---
    print(f"Loading pre-trained Flow Matcher model from {FM_MODEL_FILE}...")
    try:
        flow_model = ContinuousFlowMatcherBottleneck(
            param_dim=3, # lnA, lnf, lnfdot
            conditioning_dim=BOTTLENECK_DIM + 1 # 256 + 1
        ).to(device)
        flow_model.load_state_dict(torch.load(FM_MODEL_FILE, map_location=device))
        flow_model.eval()
        print("Successfully loaded trained Flow Matcher from Stage 2.")
    except Exception as e:
        print(f"Error loading Flow Matcher model file: {e}")
        return

    # ========================================
    # --- STAGE 3: Posterior Analysis ---
    # ========================================
    print("\n" + "="*30)
    print("--- STARTING STAGE 3: Posterior Analysis ---")
    print("="*30)
    
    # 1. Generate the "true" test signal
    custom_params_true = torch.tensor([[
        LN_CUSTOM_AMPLITUDE,
        LN_CUSTOM_FREQUENCY,
        LN_CUSTOM_FREQUENCY_DERIV
    ]], device=device).float()

    gb = GBWave(use_gpu=True)
    gb_lisa = ResponseWrapper(
        gb, 1.1 * (SIGNAL_LENGTH * DT / YRSID_SI), DT, 
        index_lambda=6, index_beta=7, t0=10000.0, flip_hx=False, use_gpu=True,
        remove_sky_coords=True, is_ecliptic_latitude=True, remove_garbage=True,
        orbits=EqualArmlengthOrbits(use_gpu=True), order=25, tdi="1st generation", tdi_chan="AET"
    )
    
    channels = gb_lisa(CUSTOM_AMPLITUDE, CUSTOM_FREQUENCY, CUSTOM_FREQUENCY_DERIV, 
                      IOTA, PHI0, PSI, LAM, BETA)
    
    # Ensure signal is correct length
    custom_signal_t = torch.tensor(channels[0].get()[:SIGNAL_LENGTH], device=device)
    custom_signal_t = custom_signal_t.reshape(1, 1, -1) # [1, 1, 518400]

    # 2. Add noise and gaps (same as in training)
    freqs = np.fft.rfftfreq(SIGNAL_LENGTH, DT)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor = torch.tensor(psd, device=device)
    
    # Use batch_size=1 for generate_noise
    noise_tensor = generate_noise(DT, psd_tensor, device=device, noise_level=1.0, N=SIGNAL_LENGTH, batch_size=1)
    noise_tensor = noise_tensor.reshape(1, 1, -1) # [1, 1, 518400]

    noisy_custom_signal = custom_signal_t + noise_tensor
    
    # Initialize mask (needed for std calculation later even if gaps are off, technically)
    # If ADD_GAPS is False, we treat all points as valid.
    if ADD_GAPS:
        print(f"Adding gaps to posterior analysis signal with lambda={GAP_LAMBDA_VALUE}")
        gaps_mask = create_gaps_mask(SIGNAL_LENGTH * DT, DT, lambda_value=GAP_LAMBDA_VALUE)
        gaps_mask_tensor = torch.from_numpy(gaps_mask).to(device).view(1, 1, -1)
        noisy_custom_signal *= gaps_mask_tensor
    else:
        gaps_mask_tensor = torch.ones_like(noisy_custom_signal)

    # 3. Standardize the signal
    # Extract scale info using gap-aware logic
    if ADD_GAPS:
        # Calculate stats on non-gap points only
        n_non_gap = gaps_mask_tensor.sum() # Scalar
        mean = (noisy_custom_signal * gaps_mask_tensor).sum() / n_non_gap
        variance = ((noisy_custom_signal - mean).pow(2) * gaps_mask_tensor).sum() / (n_non_gap - 1)
        signal_std = torch.sqrt(variance).view(1, 1, 1)
    else:
        # Standard calculation
        signal_std = torch.std(noisy_custom_signal, dim=-1, keepdim=True)
    
    # Log transform
    log_signal_std = torch.log(signal_std).squeeze()
    
    # Standardize the input signal after getting std
    noisy_custom_signal = standardize_signal(noisy_custom_signal)

    # Get bottleneck from the trained encoder
    with torch.no_grad():
        test_bottleneck = trained_encoder(noisy_custom_signal.float())  # [1, 256]
        
    print(f"Generated test bottleneck, shape: {test_bottleneck.shape}")

    # Combine bottleneck and log signal std
    test_conditioning = torch.cat([test_bottleneck, log_signal_std.view(1, -1).float()], dim=1)  # [1, 257]
    print(f"Generated test conditioning vector, shape: {test_conditioning.shape}")

    # 5. Run posterior analysis using the bottleneck
    samples, fig = analyze_posterior(
        flow_model,
        test_conditioning, # Pass the combined vector
        custom_params_true,
        params_std,
        params_mean,
        n_samples=NUM_POSTERIOR_SAMPLES,
        device=device
    )

    POSTERIOR_PLOT_FILE = os.path.join(f"outputs/run_{RUN_INDEX}", f"posterior_standalone3_{RUN_INDEX}.png")
    fig.savefig(POSTERIOR_PLOT_FILE)
    print(f"Saved posterior plot to {POSTERIOR_PLOT_FILE}")
    print("\nStage 3 Posterior Analysis completed successfully!")


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