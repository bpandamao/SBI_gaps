import torch
import numpy as np
import cupy as xp
import os
import time
import matplotlib.pyplot as plt
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support
import h5py

# Imports for the pipeline
from dataset_wavelet_sub_ite_01_update import prepare_data_from_h5 
from trainer_spectrogramV1 import train_flow_matching_spectrogram
from posterior01 import analyze_posterior
from model_utils import print_model_structure
from flow_matcher_time_asy import ContinuousFlowMatcherTime
from lisatools.sensitivity import get_sensitivity, A1TDISens
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.types import TimeSeries

torch.manual_seed(111)
np.random.seed(111)

# --- Helper functions ---
def generate_noise(dt, psd, N, device):
    """Generates noise on the specified device."""
    psd_tensor = torch.tensor(psd, device=device, dtype=torch.float32)
    freqs = torch.fft.rfftfreq(N, dt, device=device)[1:]
    variance_noise_f = N* psd_tensor / (4 * dt)
    noise_f = torch.sqrt(variance_noise_f) * (torch.randn(len(freqs), device=device) + 1j * torch.randn(len(freqs), device=device))
    noise_f = torch.cat((torch.zeros(1, device=device), noise_f))
    noise_t = torch.fft.irfft(noise_f, n=N)
    return noise_t

def transform_to_spectrogram(signal, delta_t, spec_min, spec_max):
    """
    Transforms a single 1D signal to a spectrogram and scales it
    using the provided min and max values from the training set.
    """
    Nf = 2**11
    t = np.arange(0, len(signal) * delta_t, delta_t)
    timeseries = TimeSeries(signal, t)
    wavelet_data = from_time_to_wavelet(timeseries, Nf=Nf)
    
    spectrogram = np.log(np.abs(wavelet_data.data))

    mask = (wavelet_data.freq <= 0.01) & (wavelet_data.freq >= 0.001)
    

    # Scale the spectrogram to [-1, 1] using the training set's statistics
    spectrogram_scaled = (spectrogram[mask,:] - spec_min) / (spec_max - spec_min)
    
    return torch.tensor(spectrogram_scaled, dtype=torch.float32).unsqueeze(0)


def create_gaps_mask(duration, dt, lambda_value=0.4):
    """
    Create a mask for gaps in the data using an exponential distribution.
    """
    total_points = int(duration / dt)
    mask = np.ones(total_points, dtype=np.float32)
    current_point = 0
    while current_point < total_points:
        data_length_days = np.random.exponential(scale=1/lambda_value)
        data_length_points = int(data_length_days * 24 * 3600 / dt)
        current_point += data_length_points
        if current_point >= total_points:
            break
        gap_duration_hours = np.random.uniform(0.5, 24)
        gap_duration_points = int(gap_duration_hours * 3600 / dt)
        gap_end_point = min(current_point + gap_duration_points, total_points)
        mask[current_point:gap_end_point] = 0
        current_point += gap_duration_points
    return mask

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Configuration ---
    TRAIN_DATA_FILE = "training_data/au_sp_gaps_90_trainsetvb_new_ln_smaller001_S1e5_ite_01_001_11.h5" 
    TEST_DATA_FILE = "training_data/au_sp_gaps_90_testsetvb_new_ln_smaller001_S1e3_ite_01_001_11.h5" #au_sp_gaps_90_testsetvb_ln_smaller001_S1e3_ite_01_001_11.h5"
    RUN_INDEX = "SP_gaps_update1000epoch_smaller001_asy_11bin_itev3339_90_longer_gapsnew"
    NUM_EPOCHS = 1000
    BATCH_SIZE = 128
    NUM_POSTERIOR_SAMPLES = 5000
    start_epoch = 0
    plot_start_epoch = 600
    RESUME_TRAINING = False 
    CHECKPOINT_PATH = f"outputs/run_{RUN_INDEX}/checkpoint_epoch_{start_epoch}.pt"
    # set random seed for reproducibility
    # torch.manual_seed(90)
    # np.random.seed(90)


    # Test signal parameters
    CUSTOM_AMPLITUDE = 1.5e-21
    CUSTOM_FREQUENCY = 2e-3
    CUSTOM_FREQUENCY_DERIV = 1e-10
    
    LN_CUSTOM_AMPLITUDE = np.log10(CUSTOM_AMPLITUDE)
    LN_CUSTOM_FREQUENCY = np.log(CUSTOM_FREQUENCY)
    LN_CUSTOM_FREQUENCY_DERIV = np.log(CUSTOM_FREQUENCY_DERIV)

    YRSID_SI = 31558149.763545603
    # IOTA = 0.750492  # Fixed inclination angle
    # PHI0 = 5.141845  # Fixed initial phase
    # PSI = 3.567122   # Fixed polarization angle
    # BETA = 2.973723  # Fixed ecliptic latitude
    # LAM = 5.22979888  # Fixed ecliptic longitude
    # Fixed parameters for LISA signal generation
    IOTA = 1.11820901  # Fixed inclination angle
    PHI0 = 4.91128699  # Fixed initial phase
    PSI = 2.3290324    # Fixed polarization angle
    BETA = 0.9805742971871619  # Fixed ecliptic latitude
    LAM = 5.22979888   # Fixed ecliptic longitude
    
    DT = 5.0
    DURATION = 1572864*5
    SIGNAL_LENGTH = int(DURATION / DT)

    class GBWave:
        def __init__(self, use_gpu=False): self.xp = xp if use_gpu else np
        def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
            t = self.xp.arange(0.0, T * YRSID_SI, dt)
            cos2psi, sin2psi, cosiota = self.xp.cos(2*psi), self.xp.sin(2*psi), self.xp.cos(iota)
            fddot = 11/3 * fdot**2/f
            phase = 2*np.pi*(f*t + 0.5*fdot*t**2 + 1/6*fddot*t**3) - phi0
            hSp, hSc = -self.xp.cos(phase)*A*(1+cosiota**2), -self.xp.sin(phase)*2*A*cosiota
            hp, hc = hSp*cos2psi - hSc*sin2psi, hSp*sin2psi + hSc*cos2psi
            return hp + 1j*hc


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    OUTPUT_DIR = f"outputs/run_{RUN_INDEX}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Defining parameter ranges for normalization.")
    
    # Constants for parameter ranges from the generator script
    AMPLITUDE_BASE = 1.5e-21
    FREQUENCY_BASE = 2e-3
    FREQUENCY_DERIV_BASE = 1e-10

    # True values in log space
    LN_A_TRUE = np.log10(AMPLITUDE_BASE)
    LN_F_TRUE = np.log(FREQUENCY_BASE)
    LN_FDOT_TRUE = np.log(FREQUENCY_DERIV_BASE)

    # Define min and max parameter arrays based on the known ranges
    params_min = np.array([
        LN_A_TRUE - 0.1, 
        LN_F_TRUE - 0.001, 
        LN_FDOT_TRUE - 0.001
    ])
    params_max = np.array([
        LN_A_TRUE + 0.1, 
        LN_F_TRUE + 0.001, 
        LN_FDOT_TRUE + 0.001
    ])
    
    # --- Data Loading ---
    print("Preparing data loaders from separate HDF5 files...")
    train_loader, test_loader = prepare_data_from_h5(
        train_h5_path=TRAIN_DATA_FILE, 
        test_h5_path=TEST_DATA_FILE,
        parameters_min=params_min, 
        parameters_max=params_max,
        batch_size=BATCH_SIZE, 
        num_workers=multiprocessing.cpu_count(),
        train_subset_ratio=0.3
    )
    
    # Load spectrogram scaling stats from the training file
    with h5py.File(TRAIN_DATA_FILE, 'r') as f:
        spectrogram_min = f.attrs['spectrogram_min']
        spectrogram_max = f.attrs['spectrogram_max']

    # Add 10% buffer to avoid edge issues
    buffer = 0.1 * (spectrogram_max - spectrogram_min)
    spectrogram_min -= buffer
    spectrogram_max += buffer

    print(f"Loaded spectrogram scaling stats from training data: Min={spectrogram_min:.4f}, Max={spectrogram_max:.4f}")

    np.savez(os.path.join(OUTPUT_DIR, "model_info.npz"), 
             params_min=params_min, params_max=params_max,
             spectrogram_min=spectrogram_min, spectrogram_max=spectrogram_max)
    
    model = ContinuousFlowMatcherTime(
        param_dim=3, signal_embedding_dim=512, signal_input_dim=SIGNAL_LENGTH
    ).to(device)#previous 256

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    print_model_structure(model, filename=os.path.join(OUTPUT_DIR, "model_structure.txt"))
    start_time = time.time()
    
    model, train_losses, test_losses = train_flow_matching_spectrogram(
        model, train_loader, test_loader, device,
        num_epochs=NUM_EPOCHS, lr_stage1=1e-3, lr_stage2=1e-4,
        stage1_epochs=1000, stage2_epochs=300, checkpoint_dir=None,
        start_epoch=start_epoch
    )
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    TRAINING_LOSS_FILE = os.path.join(OUTPUT_DIR, f"training_losses_{RUN_INDEX}.txt")
    LOSS_PLOT_FILE = os.path.join(OUTPUT_DIR, f"loss_history_{RUN_INDEX}.png")

    with open(TRAINING_LOSS_FILE, 'w') as f:
        f.write(f"Total training time: {training_duration:.2f} seconds\n")
        f.write("epoch,train_loss,test_loss\n")
        start_enum = start_epoch if RESUME_TRAINING else 1
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), start_enum):
            f.write(f"{epoch},{train_loss:.6f},{test_loss:.6f}\n")

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plot_start = plot_start_epoch-1 if plot_start_epoch > 0 else 0
    plt.plot(epochs[plot_start:], train_losses[plot_start:], 'b-', label='Training Loss')
    plt.plot(epochs[plot_start:], test_losses[plot_start:], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss (From Epoch {plot_start + 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT_FILE)
    plt.close()

    print(f"Training finished in {training_duration/3600:.2f} hours. Loss data and plot saved.")
    
    print("Generating posterior samples for visualization...")
    gb = GBWave(use_gpu=True)
    gb_lisa = ResponseWrapper(
        gb, 1/4, DT, 6, 7, t0=10000.0, flip_hx=False, use_gpu=True,
        remove_sky_coords=True, is_ecliptic_latitude=True, remove_garbage=True,
        orbits=EqualArmlengthOrbits(use_gpu=True), order=25, tdi="1st generation", tdi_chan="AET"
    )
    
    clean_signal_ch = gb_lisa(CUSTOM_AMPLITUDE, CUSTOM_FREQUENCY, CUSTOM_FREQUENCY_DERIV, IOTA, PHI0, PSI, LAM, BETA)
    clean_signal = clean_signal_ch[0].get()[:SIGNAL_LENGTH]
    
    freqs = np.fft.rfftfreq(SIGNAL_LENGTH, DT)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    noise = generate_noise(DT, psd, N=SIGNAL_LENGTH, device='cpu').numpy()
    
    noisy_signal = clean_signal + noise

    gaps_mask = create_gaps_mask(DURATION, DT)
    gapped_noisy_signal = noisy_signal * gaps_mask
    print(f"Applying a random gap mask to posterior signal (Duty Cycle: {np.mean(gaps_mask):.2f})")

    # Transform signal to spectrogram using training set scaling stats
    noisy_spectrogram = transform_to_spectrogram(
        gapped_noisy_signal, DT, spectrogram_min, spectrogram_max
    ).to(device)

    _, fig = analyze_posterior(
        model, noisy_spectrogram, torch.tensor([[LN_CUSTOM_AMPLITUDE, LN_CUSTOM_FREQUENCY, LN_CUSTOM_FREQUENCY_DERIV]]),
        params_min, params_max, n_samples=NUM_POSTERIOR_SAMPLES, device=device
    )
    
    fig.savefig(os.path.join(OUTPUT_DIR, "posterior.png"))
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "flow_matcher.pt"))
    print("Posterior plot and final model saved.")

if __name__ == "__main__":
    freeze_support()
    main()

