import numpy as np
import os
import time
import h5py
from tqdm import tqdm
from lisatools.sensitivity import get_sensitivity, A1TDISens
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Imports for Wavelet Transform
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.types import TimeSeries

# Configuration
# Modify these constants before running
INPUT_DATA_FILE = "training_data/fullsignal_training_set_TIME_ln_S2e4_90d_smaller0010005_range_trainnew_ln.npz"
OUTPUT_H5_FILE = "training_data/au_sp_gaps_90_trainsetvb_ln_smaller0010005_S1e5_ite_01_001_11.h5"
examplefilename = "example_spectrogram_optimized_90dtrain0010005.png"
NOISE_AUGMENTATIONS = 5  # Number of different noise/gap instances per signal
N_JOBS = -3  # Number of parallel jobs (-1 = all cores, negative = all but N cores)

def create_gaps_mask(duration, dt, lambda_value=0.75):
    """
    Create a mask for gaps in the data using an exponential distribution.
    A simple model where the instrument is on for a period, then off for a period.
    
    Args:
        duration (float): Total duration of the signal in seconds.
        dt (float): The time step (sampling interval).
        lambda_value (float): Parameter for the exponential distribution to model segment length.
        
    Returns:
        np.ndarray: A binary mask with 0s for gaps and 1s for data.
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

        gap_duration_hours = np.random.uniform(3.5, 12)# 3.5 12
        gap_duration_points = int(gap_duration_hours * 3600 / dt)

        gap_end_point = min(current_point + gap_duration_points, total_points)
        mask[current_point:gap_end_point] = 0
        
        current_point += gap_duration_points

    return mask

def transform_to_spectrogram(signal, delta_t):
    """
    Takes a single 1D signal, performs a CWT, and returns the
    log-absolute value of the transform data.
    """
    Nf = 2**11  # Number of wavelet frequency bins
    t = np.arange(0, len(signal) * delta_t, delta_t)
    timeseries = TimeSeries(signal, t)
    
    wavelet_data_obj = from_time_to_wavelet(timeseries, Nf=Nf)

    # subset freq between 0.005 to 0.001
    mask = (wavelet_data_obj.freq <=0.01) & (wavelet_data_obj.freq >=0.001)

    # Return the raw log-spectrogram and the wavelet object for plotting
    unscaled_log_spectrogram = np.log(np.abs(wavelet_data_obj.data[mask,:])).astype(np.float32)

    return wavelet_data_obj, unscaled_log_spectrogram

def plot_wavelet(signal_wavelet_freq, output_path, freq_range=None):
    """
    Plots a wavelet spectrogram and saves it to a file.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    signal_wavelet_freq.plot(ax=ax, zscale="log", absolute=True, freq_range=[0.001, 0.01] if freq_range is None else freq_range)
    ax.set_title('Wavelet Transform of Noisy Signal (log scale)', fontsize=16)
    plt.savefig(output_path)
    plt.clf()
    plt.close(fig)
    print(f"Saved example wavelet plot to {output_path}")

def generate_noise_cpu(dt, psd, N):
    """
    Generates noise on the CPU using numpy.
    """
    freqs = np.fft.rfftfreq(N, dt)[1:]
    # N_t = int(2**(np.ceil(np.log2(N))))
    variance_noise_f = N*psd / (4 * dt)
    
    noise_f_real = np.random.randn(len(freqs))
    noise_f_imag = np.random.randn(len(freqs))
    noise_f = np.sqrt(variance_noise_f) * (noise_f_real + 1j * noise_f_imag)
    
    noise_f = np.concatenate((np.zeros(1), noise_f))
    noise_t = np.fft.irfft(noise_f,N)
    return noise_t


def process_signal(clean_signal, params, dt, psd, signal_length, signal_duration):
    """
    This function takes one clean signal and generates all its noisy augmentations.
    It's designed to be called in parallel.
    It returns the original parameters once, and a list of all spectrograms
    for that signal, along with their duty cycles.
    """
    augmented_spectrograms_for_this_signal = []
    duty_cycles_local = []
    
    for _ in range(NOISE_AUGMENTATIONS):
        noise = generate_noise_cpu(dt, psd, N=signal_length)
        noisy_signal = clean_signal + noise
        
        gaps_mask = create_gaps_mask(signal_duration, dt)
        duty_cycles_local.append(np.mean(gaps_mask))
        
        gapped_noisy_signal = noisy_signal * gaps_mask
        
        _, unscaled_log_spectrogram = transform_to_spectrogram(gapped_noisy_signal, dt)
        
        augmented_spectrograms_for_this_signal.append(unscaled_log_spectrogram)
        
    return params, augmented_spectrograms_for_this_signal, duty_cycles_local


def augment_and_transform_parallel():
    # Load the original 1D signal data
    output_folder = "training_data"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading clean 1D signals from {INPUT_DATA_FILE}...")
    try:
        data = np.load(INPUT_DATA_FILE)
        original_parameters = data['parameters']
        original_time_signals = data['time_signals']
        time_array = data['time_array']
        dt = time_array[1] - time_array[0]
        signal_length = original_time_signals.shape[1]
        signal_duration = signal_length * dt
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    num_original_samples = original_time_signals.shape[0]
    print(f"Loaded {num_original_samples} signals of length {signal_length}.")

    freqs_psd = np.fft.rfftfreq(signal_length, dt)[1:]
    psd = get_sensitivity(freqs_psd, sens_fn=A1TDISens, return_type="PSD")
    
    # --- Plot one example before starting the heavy processing ---
    print("Generating one example spectrogram plot...")
    clean_signal_example = original_time_signals[0]
    noise_example = generate_noise_cpu(dt, psd, N=signal_length)
    gaps_mask_example = create_gaps_mask(signal_duration, dt)
    gapped_noisy_signal_example = (clean_signal_example + noise_example) * gaps_mask_example
    unscaled_wavelet_obj, _ = transform_to_spectrogram(gapped_noisy_signal_example, dt)
    plot_wavelet(unscaled_wavelet_obj, os.path.join(output_folder, examplefilename))

    # Process data in parallel
    n_cores_str = 'all' if N_JOBS == -1 else str(abs(N_JOBS))
    print(f"Starting parallel generation of {num_original_samples * NOISE_AUGMENTATIONS} spectrograms using {n_cores_str} cores...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_signal)(
            original_time_signals[i], 
            original_parameters[i], 
            dt, psd, signal_length, signal_duration
        ) for i in tqdm(range(num_original_samples), desc="Processing original signals")
    )
    
    # Unpack and save results
    print("\nParallel processing complete. Consolidating and saving results...")
    all_params, all_spectrograms_list, all_duty_cycles_list = zip(*results)
    
    final_original_params = np.array(all_params)
    final_spectrograms = np.array(all_spectrograms_list)
    duty_cycles = [item for sublist in all_duty_cycles_list for item in sublist]

    # Calculate and save spectrogram min/max for scaling
    spectrogram_min = np.min(final_spectrograms)
    spectrogram_max = np.max(final_spectrograms)
    print(f"\nCalculated Spectrogram Stats for Scaling: Min={spectrogram_min:.4f}, Max={spectrogram_max:.4f}")

    with h5py.File(OUTPUT_H5_FILE, 'w') as hf:
        hf.create_dataset('parameters', data=final_original_params, dtype='f4')
        hf.create_dataset('spectrograms', data=final_spectrograms, dtype='f4')
        # Save stats as attributes of the HDF5 file
        hf.attrs['spectrogram_min'] = spectrogram_min
        hf.attrs['spectrogram_max'] = spectrogram_max

    print("\n--- Data augmentation complete ---")
    print(f"Total original samples: {num_original_samples}")
    print(f"Total augmented spectrograms generated: {num_original_samples * NOISE_AUGMENTATIONS}")
    print(f"Data saved to: {OUTPUT_H5_FILE}")
    
    # Plot duty cycle distribution
    plt.figure(figsize=(10, 6))
    plt.hist(duty_cycles, bins=50, density=True)
    plt.title('Distribution of On-Duty Cycles')
    plt.xlabel('Duty Cycle')
    plt.ylabel('Density')
    plt.grid(True)
    duty_cycle_plot_path = os.path.join(output_folder, "duty_cycle_distribution_optimized_iterative.png")
    plt.savefig(duty_cycle_plot_path)
    plt.close()
    print(f"Saved duty cycle distribution plot to {duty_cycle_plot_path}")
    print("---------------------------------")

if __name__ == "__main__":
    start_time = time.time()
    augment_and_transform_parallel()
    end_time = time.time()
    print(f"\nTotal script time: {end_time - start_time:.2f} seconds")

