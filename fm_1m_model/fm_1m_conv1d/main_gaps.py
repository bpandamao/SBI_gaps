import torch
import numpy as np
import cupy as xp
from posterior import analyze_posterior
from dataset_time import prepare_data, generate_noise
# --- MODIFIED: Import from trainer_gaps ---
from trainer_gaps import train_flow_matching, create_gaps_mask
from model_utils import print_model_structure
from flow_matcher_time import ContinuousFlowMatcherTime
import matplotlib.pyplot as plt
import time
import os
from lisatools.sensitivity import get_sensitivity, A1TDISens
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass

# --- MODIFIED: Constants for Gaps ---
DATA_INDEX = "ln_30_DAY"
# Append "_gaps" to the run index to distinguish this run
RUN_INDEX = "TIME_FM_ln_gaps_new" 
ADD_GAPS = True  # Set to True to enable training with gaps
# Lambda value for gap creation (0.6 to 0.85 is a good range)
GAP_LAMBDA_VALUE = 0.75 

NUM_EPOCHS = 300 # Total number of epochs
BATCH_SIZE = 128
NUM_POSTERIOR_SAMPLES = 5000
start_epoch = 0
RESUME_TRAINING = False
CHECKPOINT_PATH = f"outputs/run_{RUN_INDEX}/checkpoint_epoch_{start_epoch}.pt"

# Add custom parameter constants (for posterior analysis)
CUSTOM_AMPLITUDE = 5e-21
CUSTOM_FREQUENCY = 2e-3
CUSTOM_FREQUENCY_DERIV = 3e-10

LN_CUSTOM_AMPLITUDE = np.log(CUSTOM_AMPLITUDE)
LN_CUSTOM_FREQUENCY = np.log(CUSTOM_FREQUENCY)
LN_CUSTOM_FREQUENCY_DERIV = np.log(CUSTOM_FREQUENCY_DERIV)

# LN_CUSTOM_AMPLITUDE = np.log(CUSTOM_AMPLITUDE)
# LN_CUSTOM_FREQUENCY = np.log(CUSTOM_FREQUENCY)
# LN_CUSTOM_FREQUENCY_DERIV = np.log10(CUSTOM_FREQUENCY_DERIV)

# Constants from training_data_generator_time.py
YRSID_SI = 31558149.763545603
IOTA = 1.11820901
PHI0 = 4.91128699
PSI = 2.3290324
BETA = 0.9805742971871619
LAM = 5.22979888

class GBWave:
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
    """
    Standardize a signal by subtracting mean and dividing by std.
    Add a small epsilon to avoid division by zero for silent signals (gaps).
    """
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    return (signal - mean) / (std)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

OUTPUT_DIR = f"outputs/run_{RUN_INDEX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_INFO_FILE = os.path.join(OUTPUT_DIR, f"model_info_{RUN_INDEX}.npz")
MODEL_STRUCTURE_FILE = os.path.join(OUTPUT_DIR, f"model_structure_{RUN_INDEX}.txt")
TRAINING_LOSS_FILE = os.path.join(OUTPUT_DIR, f"training_losses_{RUN_INDEX}.txt")
LOSS_PLOT_FILE = os.path.join(OUTPUT_DIR, f"loss_history_{RUN_INDEX}.png")
MODEL_STATE_FILE = os.path.join(OUTPUT_DIR, f"flow_matcher_{RUN_INDEX}.pt")
POSTERIOR_PLOT_FILE = os.path.join(OUTPUT_DIR, f"posterior_{RUN_INDEX}.png")

try:
    data = np.load(f'training_data/training_set_{DATA_INDEX}.npz')
    parameters = data['parameters']
    time_signals = data['time_signals']
    time_array = data['time_array']
except Exception as e:
    print(f"Error loading training data: {e}")
    print("Please run training_data_generator_time.py first to generate the training data.")
    exit()

try:
    time_signals_reshaped = time_signals.reshape(time_signals.shape[0], 1, time_signals.shape[1])
    train_loader, test_loader, params_mean, params_std = prepare_data(
        parameters, 
        time_signals_reshaped,
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"Error preparing data: {e}")
    exit()

np.savez(MODEL_INFO_FILE, params_mean=params_mean, params_std=params_std)

model = ContinuousFlowMatcherTime(
    param_dim=3,
    hidden_dim=256,
    signal_input_dim=time_signals.shape[1]
).to(device)

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Successfully loaded checkpoint")
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Starting training from scratch")
else:
    print("Starting training from scratch")

print_model_structure(model, filename=MODEL_STRUCTURE_FILE)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.set_num_threads(multiprocessing.cpu_count() - 1)

    start_time = time.time()
    
    # --- MODIFIED: Pass gap parameters to the trainer ---
    model, train_losses, test_losses = train_flow_matching(
        model, 
        train_loader, 
        test_loader, 
        device,
        num_epochs=NUM_EPOCHS,
        curriculum_epochs=200,
        max_noise_level=1.0,
        checkpoint_dir=OUTPUT_DIR,
        start_epoch=start_epoch if RESUME_TRAINING else 0,
        warmup_epochs=0,
        lr_stage1=1e-4,
        lr_stage2=1e-5,
        stage1_epochs=250,
        stage2_epochs=50,
        new_lr_multiplier_stage1=1,
        new_lr_multiplier_stage2=1,
        eta_min_ratio=10,
        num_workers=multiprocessing.cpu_count() - 1,
        # Pass gap arguments
        add_gaps=ADD_GAPS,
        gap_lambda_value=GAP_LAMBDA_VALUE
    )

    end_time = time.time()
    training_duration = end_time - start_time

    with open(TRAINING_LOSS_FILE, 'w') as f:
        f.write(f"Total training time: {training_duration:.2f} seconds\n")
        f.write("epoch,train_loss,test_loss\n")
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), 1):
            f.write(f"{epoch},{train_loss:.6f},{test_loss:.6f}\n")

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plot_start = 0 if len(train_losses) <= start_epoch else start_epoch
    plt.plot(epochs[plot_start:], train_losses[plot_start:], 'b-', label='Training Loss')
    plt.plot(epochs[plot_start:], test_losses[plot_start:], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss (After Epoch {plot_start})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT_FILE)
    plt.close()

    custom_params = torch.tensor([[
        LN_CUSTOM_AMPLITUDE,
        LN_CUSTOM_FREQUENCY,
        LN_CUSTOM_FREQUENCY_DERIV
    ]], device=device).float()

    T = 1/12
    dt = 5
    order = 25
    tdi_gen = "1st generation"
    index_lambda = 6
    index_beta = 7
    t0 = 10000.0
    use_gpu=True
    gb = GBWave(use_gpu=True)
    
    gb_lisa = ResponseWrapper(
        gb, T, dt, index_lambda, index_beta, t0=t0, flip_hx=False, use_gpu=use_gpu,
        remove_sky_coords=True, is_ecliptic_latitude=True, remove_garbage=True,
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AET"
    )

    duration = 30 * 24 * 3600
    signal_length = int(duration / dt)

    channels = gb_lisa(CUSTOM_AMPLITUDE, CUSTOM_FREQUENCY, CUSTOM_FREQUENCY_DERIV, 
                      IOTA, PHI0, PSI, LAM, BETA)
    custom_signal_t = torch.tensor(channels[0][:signal_length], device=device)

    freqs = np.fft.rfftfreq(signal_length, dt)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor =torch.tensor(psd, device=device)

    noise = generate_noise(dt, psd_tensor, device=device)
    noise_tensor = noise[:signal_length].reshape(1, 1, -1)

    custom_test_signal = custom_signal_t.reshape(1, 1, -1)
    noisy_custom_signal = custom_test_signal + noise_tensor
    
    # Apply gaps to the test signal for posterior analysis
    if ADD_GAPS:
        print(f"Adding gaps to posterior analysis signal with lambda={GAP_LAMBDA_VALUE}")
        gaps_mask = create_gaps_mask(duration, dt, lambda_value=GAP_LAMBDA_VALUE)
        print(f"duty cycle of gaps: {np.mean(gaps_mask):.2f}")
        gaps_mask_tensor = torch.from_numpy(gaps_mask).to(device)
        noisy_custom_signal *= gaps_mask_tensor.view(1, 1, -1)
        
    # show gapped signal
    plt.figure(figsize=(10, 6))
    plt.plot(noisy_custom_signal[0, 0].cpu().numpy(), label='Noisy Custom Signal with Gaps')
    plt.title('Noisy Custom Signal with Gaps')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"noisy_custom_signal_with_gaps_{RUN_INDEX}.png"))
    plt.close() 


    # Standardize the final signal before passing to the model
    mean = torch.mean(noisy_custom_signal, dim=-1, keepdim=True)
    std = torch.std(noisy_custom_signal, dim=-1, keepdim=True)
    noisy_custom_signal = (noisy_custom_signal - mean) / std

    samples, fig = analyze_posterior(
        model,
        noisy_custom_signal,
        custom_params,
        params_std,
        params_mean,
        n_samples=NUM_POSTERIOR_SAMPLES,
        device=device
    )

    try:
        torch.save(model.state_dict(), MODEL_STATE_FILE)
        fig.savefig(POSTERIOR_PLOT_FILE)
    except Exception as e:
        print(f"Error saving results: {e}")
        exit()

    print("Training completed successfully!")

    if __name__ == "__main__":
        freeze_support()
