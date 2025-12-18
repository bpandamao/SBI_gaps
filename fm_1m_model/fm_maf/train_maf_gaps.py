"""
Training script for Masked Autoregressive Flow (MAF) model with gaps support.
"""

import torch
import numpy as np
import cupy as xp
import matplotlib.pyplot as plt
import corner
import os
import time
from tqdm import tqdm

from model_maf_gaps_new import create_maf_model
from dataset_time import prepare_data, generate_noise
from lisatools.sensitivity import get_sensitivity, A1TDISens
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits


class GBWave:
    def __init__(self, use_gpu=False):
        self.xp = xp if use_gpu else np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0, YRSID_SI=31558149.76):
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)
        fddot = 11.0 / 3.0 * fdot ** 2 / f
        phase = 2 * np.pi * (f * t + 0.5 * fdot * t**2 + (1/6) * fddot * t**3) - phi0
        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota**2)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hc

def create_gaps_mask(duration, dt, lambda_value=0.75):
    """Create a mask for gaps in the data using an exponential distribution."""
    total_points = int(duration / dt)
    mask = np.ones(total_points, dtype=np.float32)
    current_point = 0
    while current_point < total_points:
        data_length_days = np.random.exponential(scale=1/lambda_value)
        data_length_points = int(data_length_days * 24 * 3600 / dt)
        current_point += data_length_points
        if current_point >= total_points: break
        gap_duration_hours = np.random.uniform(3.5, 12)
        gap_duration_points = int(gap_duration_hours * 3600 / dt)
        gap_end_point = min(current_point + gap_duration_points, total_points)
        mask[current_point:gap_end_point] = 0
        current_point += gap_duration_points
    return mask

def standardize_signal(signal):
    """Standardize a signal by subtracting mean and dividing by std."""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    return (signal - mean) / (std) # Add epsilon for stability

def get_model_structure(model):
    """Get model structure as a formatted string."""
    lines = [f"{'Model Structure':-^50}"]
    lines.append(str(model))
    lines.append(f"\n{'Trainable Parameters':-^50}")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            lines.append(f"{name:<40s} | Shape: {list(param.shape)} | Params: {param_count:,}")
    lines.append("-" * 50)
    lines.append(f"Total Trainable Parameters: {total_params:,}")
    return "\n".join(lines)

def train_loop(model, dataloader, optimizer, device, psd_tensor, delta_t, duration, add_gaps, gap_lambda_value, current_noise_level):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for signal, target in pbar:
        optimizer.zero_grad()
        signal, target = signal.to(device, non_blocking=True).float(), target.to(device, non_blocking=True).float()
        batch_size, _, signal_length = signal.shape
        
        # Generate noise sample by sample
        noise_tensor = torch.zeros_like(signal, device=device)
        for i in range(batch_size):
            noise = generate_noise(delta_t, psd_tensor, device=device, noise_level=current_noise_level, N=signal_length)
            noise_tensor[i, 0, :] = noise[:signal_length]

        noisy_signal = signal + noise_tensor

        if add_gaps:
            for i in range(batch_size):
                gaps_mask = create_gaps_mask(duration, delta_t, lambda_value=gap_lambda_value)
                noisy_signal[i, 0, :] *= torch.from_numpy(gaps_mask).to(device)
        
        # Squeeze the channel dimension for the context processor
        final_signal = standardize_signal(noisy_signal.squeeze(1))
        loss = -model.log_prob(target, context=final_signal).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

def test_loop(model, dataloader, device, psd_tensor, delta_t, duration, add_gaps, gap_lambda_value, current_noise_level):
    """Validation loop for one epoch."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for signal, target in dataloader:
            signal, target = signal.to(device, non_blocking=True).float(), target.to(device, non_blocking=True).float()
            batch_size, _, signal_length = signal.shape

            # Generate noise sample by sample
            noise_tensor = torch.zeros_like(signal, device=device)
            for i in range(batch_size):
                noise = generate_noise(delta_t, psd_tensor, device=device, noise_level=current_noise_level, N=signal_length)
                noise_tensor[i, 0, :] = noise[:signal_length]

            noisy_signal = signal + noise_tensor

            if add_gaps:
                for i in range(batch_size):
                    gaps_mask = create_gaps_mask(duration, delta_t, lambda_value=gap_lambda_value)
                    noisy_signal[i, 0, :] *= torch.from_numpy(gaps_mask).to(device)
            
            # Squeeze the channel dimension for the context processor
            final_signal = standardize_signal(noisy_signal.squeeze(1))
            loss = -model.log_prob(target, context=final_signal).mean()
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # --- Configuration ---
    DATA_INDEX = "ln_30_DAY"
    RUN_INDEX = "MAF_flexible_gaps_ln" 
    OUTPUT_DIR = f"outputs/MAF_run_{RUN_INDEX}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training settings
    ADD_GAPS = True
    GAP_LAMBDA_VALUE = 0.75
    NUM_EPOCHS = 250
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    
    # Model Architecture Settings
    HIDDEN_DIMS_LIST = [256, 128, 64]  # List of hidden dims for each MAF block
    NUM_TRANSFORM_BLOCKS = 2          # Number of residual blocks within each MAF
    CONTEXT_EMBEDDING_DIM = 256       # Dimension of the context embedding from the CNN
    USE_BATCH_NORM = True
    
    # Noise curriculum
    MAX_NOISE_LEVEL = 1.0
    CURRICULUM_EPOCHS = 200

    # Posterior analysis
    NUM_POSTERIOR_SAMPLES = 5000
    LN_CUSTOM_AMPLITUDE = np.log(5e-21)
    LN_CUSTOM_FREQUENCY = np.log(2e-3)
    LN_CUSTOM_FREQUENCY_DERIV = np.log(3e-10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Data ---
    data_path = f'../training_data/training_set_{DATA_INDEX}.npz'
    data = np.load(data_path)
    parameters, time_signals, time_array = data['parameters'], data['time_signals'], data['time_array']
    delta_t, duration = time_array[1] - time_array[0], time_array[-1] + (time_array[1] - time_array[0])
    print(f"Loaded {len(parameters)} signals. Duration: {duration/86400:.1f} days, dt: {delta_t}s")

    # --- Prepare DataLoaders ---
    time_signals_reshaped = time_signals.reshape(time_signals.shape[0], 1, time_signals.shape[1])
    train_loader, test_loader, params_mean, params_std = prepare_data(parameters, time_signals_reshaped, batch_size=BATCH_SIZE)
    np.savez(os.path.join(OUTPUT_DIR, f"params_stats_{RUN_INDEX}.npz"), params_mean=params_mean, params_std=params_std)

    # --- Setup Model ---
    param_dim, context_dim = parameters.shape[1], time_signals.shape[1]
    
    # Create MAF model
    model = create_maf_model(
        param_dim=param_dim,
        context_dim=context_dim,
        hidden_dims=HIDDEN_DIMS_LIST,
        num_transform_blocks=NUM_TRANSFORM_BLOCKS,
        batch_norm=USE_BATCH_NORM,
        context_embedding_dim=CONTEXT_EMBEDDING_DIM
    ).to(device)
    
    model_structure_info = get_model_structure(model)
    print(model_structure_info)
    with open(os.path.join(OUTPUT_DIR, f"model_structure_{RUN_INDEX}.txt"), 'w') as f:
        f.write(model_structure_info)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Setup Noise Generation ---
    signal_length = time_signals.shape[1]
    freqs = np.fft.rfftfreq(signal_length, delta_t)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor = torch.tensor(psd, device=device, dtype=torch.float32)

    # --- Training Loop ---
    train_losses, test_losses = [], []
    start_time = time.time()
    main_pbar = tqdm(range(NUM_EPOCHS), desc="Total Progress")
    for epoch in main_pbar:
        current_noise_level = MAX_NOISE_LEVEL * min(1.0, (epoch + 1) / CURRICULUM_EPOCHS)
        train_loss = train_loop(model, train_loader, optimizer, device, psd_tensor, delta_t, duration, ADD_GAPS, GAP_LAMBDA_VALUE, current_noise_level)
        test_loss = test_loop(model, test_loader, device, psd_tensor, delta_t, duration, ADD_GAPS, GAP_LAMBDA_VALUE, current_noise_level)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        main_pbar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}", noise=f"{current_noise_level:.2f}")

    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # --- Save Results ---
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{RUN_INDEX}.pt'))
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Negative Log-Likelihood'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True); plt.yscale('log')
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss_history_{RUN_INDEX}.png'))
    plt.close()

    # --- Posterior Analysis ---
    print("Running posterior analysis...")
    YRSID_SI = 31558149.763545603
    T_MAX = 30*3600*24  # 1 year in seconds
    T_obs = 1.1 * (T_MAX / YRSID_SI)
    IOTA, PHI0, PSI = 1.11820901, 4.91128699, 2.3290324
    LAM, BETA = 5.22979888, 0.9805742971871619
    gb = GBWave(use_gpu=True)
    gb_lisa = ResponseWrapper(gb, duration/YRSID_SI, delta_t, 6, 7, t0=10000.0, flip_hx=False, use_gpu=True, remove_sky_coords=True, is_ecliptic_latitude=True, remove_garbage=True, orbits=EqualArmlengthOrbits(use_gpu=True), order=25, tdi="1st generation", tdi_chan="AET")
    
    # Generate test signal
    custom_signal_gpu = gb_lisa(np.exp(LN_CUSTOM_AMPLITUDE), np.exp(LN_CUSTOM_FREQUENCY), np.exp(LN_CUSTOM_FREQUENCY_DERIV), IOTA, PHI0, PSI, LAM, BETA)[0]
    custom_signal_t = torch.from_numpy(xp.asnumpy(custom_signal_gpu[:signal_length])).to(device, dtype=torch.float32)

    # Generate noise for a single sample
    noise = generate_noise(delta_t, psd_tensor, device=device, noise_level=MAX_NOISE_LEVEL, N=signal_length)
    noisy_custom_signal = custom_signal_t + noise[:signal_length]
    
    if ADD_GAPS:
        gaps_mask = create_gaps_mask(duration, delta_t, lambda_value=GAP_LAMBDA_VALUE)
        noisy_custom_signal *= torch.from_numpy(gaps_mask).to(device)

    final_test_signal = standardize_signal(noisy_custom_signal)

    model.eval()
    with torch.no_grad():
        samples = model.sample(NUM_POSTERIOR_SAMPLES, context=final_test_signal.unsqueeze(0)).squeeze(0).cpu().numpy()

    samples_denorm = samples * params_std + params_mean
    true_params = np.array([LN_CUSTOM_AMPLITUDE, LN_CUSTOM_FREQUENCY, LN_CUSTOM_FREQUENCY_DERIV])

    fig = corner.corner(samples_denorm, labels=[r"$\ln(A)$", r"$\ln(f)$", r"$\ln(\dot{f})$"], truths=true_params, show_titles=True, title_fmt=".2e", quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 12})
    fig.suptitle(f"Posterior for {RUN_INDEX}", fontsize=16)
    fig.savefig(os.path.join(OUTPUT_DIR, f'corner_plot_{RUN_INDEX}.png'))
    plt.close()

    print(f"Evaluation complete! Results saved in '{OUTPUT_DIR}' directory.")
