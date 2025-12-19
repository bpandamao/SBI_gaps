import numpy as np
import os
import time
import cupy as xp
# from scipy.signal import welch
# from lisatools.sensitivity import get_sensitivity, A1TDISens
# from fastlisaresponse import ResponseWrapper
# from lisatools.detector import EqualArmlengthOrbits
import h5py
from fastlisaresponse import pyResponseTDI, ResponseWrapper
# from astropy import units as un

from lisatools.detector import EqualArmlengthOrbits, ESAOrbits

# Constants
RUN_ID = "training_set_90d"
AMPLITUDE_BASE = 1.5e-21
# AMPLITUDE_SCALE = 1e-21
FREQUENCY_BASE = 2e-3
# FREQUENCY_SCALE = 9e-5
FREQUENCY_DERIV_BASE = 1e-10
# FREQUENCY_DERIV_SCALEa = 5e-10
# FREQUENCY_DERIV_SCALEb = 1e-15
T_MAX = 93*24*3600  # 90 days in seconds
dt = 5  # 5 seconds sampling
NUM_SAMPLES = 10000 # 20000
SIGNAL_INPUT_SIZE = 1572864  # Number of time steps (90d * 24h * 60m * 60s / 5s = 518400)
YRSID_SI = 31558149.763545603

# Fixed parameters for LISA signal generation
IOTA = 1.11820901  # Fixed inclination angle
PHI0 = 4.91128699  # Fixed initial phase
PSI = 2.3290324    # Fixed polarization angle
BETA = 0.9805742971871619  # Fixed ecliptic latitude
LAM = 5.22979888   # Fixed ecliptic longitude
full_year_points= 6307629

class GBWave:
    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=5.0):
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
    


# # Your time domain signal
# signal_A_td = A_channel
# dt = 5

# # Convert to frequency domain
# signal_A_fd = np.fft.fft(signal_A_td)
# frequencies = np.fft.fftfreq(len(signal_A_td), dt)
# frequencies = frequencies[:len(frequencies)//2]  # Only positive frequencies
# signal_A_fd = signal_A_fd[:len(frequencies)][1:]

# # Your noise PSD
# noise_psd =  get_sensitivity(frequencies[1:], sens_fn=A1TDISens, return_type="PSD")# Should match the frequency array

# # Calculate SNR manually
# # SNR^2 = 4 * sum(|h(f)|^2 / S_n(f)) * df
# df = frequencies[1] - frequencies[0]  # Frequency resolution

# # Only use positive frequencies
# signal_power = np.abs(signal_A_fd)**2
# snr_squared = 4 * df * np.sum(signal_power / noise_psd)
# snr = np.sqrt(snr_squared)

# print(f"SNR: {snr}")

def generate_training_set():
    # Create output folder if it doesn't exist
    output_folder = "training_data"
    os.makedirs(output_folder, exist_ok=True)
    
    # # Generate only a, f, fdot parameters
    # a = (AMPLITUDE_BASE + AMPLITUDE_SCALE * np.random.uniform(-1, 1, NUM_SAMPLES))
    # f = (FREQUENCY_BASE + FREQUENCY_SCALE * np.random.uniform(-1, 1, NUM_SAMPLES))
    # fdot = (FREQUENCY_DERIV_BASE + FREQUENCY_DERIV_SCALE * np.random.uniform(-1, 1, NUM_SAMPLES))

    # Define true values in linear space
    A_TRUE = AMPLITUDE_BASE
    F_TRUE = FREQUENCY_BASE
    FDOT_TRUE = FREQUENCY_DERIV_BASE

    # Convert true values to log space
    LN_A_TRUE = np.log10(A_TRUE)
    LN_F_TRUE = np.log(F_TRUE)
    LN_FDOT_TRUE = np.log(FDOT_TRUE)
    # LN_FREQUENCY_DERIV_SCALEa = np.log10(FREQUENCY_DERIV_SCALEa)
    # LN_FREQUENCY_DERIV_SCALEb = np.log10(FREQUENCY_DERIV_SCALEb)

    # Define ranges in log space
    LN_A_RANGE = [LN_A_TRUE - 0.1, LN_A_TRUE + 0.1]#0.05 #0.1 c, 0.05 a
    LN_F_RANGE = [LN_F_TRUE - 0.002, LN_F_TRUE + 0.002] #0.00005#0.00001 for smallest 0.0001 smaller
    LN_FDOT_RANGE = [LN_FDOT_TRUE - 0.001, LN_FDOT_TRUE + 0.001] #0.00001for smallest

    # Generate parameters by sampling uniformly in log space
    ln_a = np.random.uniform(LN_A_RANGE[0], LN_A_RANGE[1], NUM_SAMPLES)
    ln_f = np.random.uniform(LN_F_RANGE[0], LN_F_RANGE[1], NUM_SAMPLES)
    # ln_fdot = np.random.uniform(np.log(FREQUENCY_DERIV_SCALEb),np.log(FREQUENCY_DERIV_SCALEa) , NUM_SAMPLES)
    # ln_fdot = np.log(fdot)
    ln_fdot= np.random.uniform(LN_FDOT_RANGE[0], LN_FDOT_RANGE[1], NUM_SAMPLES)

    # Convert the log-space parameters back to linear space for waveform generation
    a = 10**(ln_a)
    f = np.exp(ln_f)
    fdot = np.exp(ln_fdot)

    parameters = np.vstack((ln_a, ln_f, ln_fdot)).T
    
    # parameters = np.vstack((a, f, fdot)).T
    
    # Setup LISA response wrapper
    T = 1/4 # Convert to years
    order = 25
    tdi_gen = "1st generation"
    index_lambda = 6
    index_beta = 7
    t0 = 10000.0
    use_gpu = True
    
    tdi_kwargs_esa = dict(
        order=order, 
        tdi=tdi_gen, 
        tdi_chan="AET",
    )
    
    # Initialize waveform generator and response wrapper
    gb = GBWave(use_gpu=use_gpu)
    gb_lisa = ResponseWrapper(
        gb,
        T,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=False,
        use_gpu=use_gpu,
        remove_sky_coords=True,
        is_ecliptic_latitude=True,
        remove_garbage=True,
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        **tdi_kwargs_esa,
    )
    
    # Generate time-domain signals
    time_signals = np.empty((NUM_SAMPLES, SIGNAL_INPUT_SIZE))

    # Calculate full year and last 30 days in seconds
    # days_30_seconds = 30 * 24 * 60 * 60
    

    # # full_year_points = len(channels[0])
    days_90_points = SIGNAL_INPUT_SIZE
    
    for i in range(NUM_SAMPLES):
        # Generate TDI channels using fixed parameters for iota, phi0, psi, beta, lam
        channels = gb_lisa(a[i], f[i], fdot[i], IOTA, PHI0, PSI, LAM, BETA)
        
        # Extract A channel (first channel) the last 90 days
        time_signals[i] = channels[0].get()[:days_90_points]
        
        if (i+1)%10==0:
            print(f"{i+1} sets have been generated!")
    
    # full_year_points = len(channels[0])
    # print
    # # store the index of the last 30 days
    # last_30_days_index = full_year_points - days_30_points - 1

    # Save all necessary data
    save_name = os.path.join(output_folder, f"fullsignal_{RUN_ID}")
    np.savez(f'{save_name}.npz',
             parameters=parameters,
             time_signals=time_signals,
             time_array=np.arange(0, days_90_points * dt, dt))
    
    print(f"\nGenerated training set with {NUM_SAMPLES} samples")
    print(f"Saved to '{save_name}.npz'")
    print("\nParameter ranges:")
    print(f"log_Amplitude: [{parameters[:,0].min():.2e}, {parameters[:,0].max():.2e}]")
    print(f"log_Frequency: [{parameters[:,1].min():.5e}, {parameters[:,1].max():.5e}]")
    print(f"log_Frequency derivative: [{parameters[:,2].min():.5e}, {parameters[:,2].max():.5e}]")
    print(f"Amplitude: [{10**(parameters[:,0]).min():.2e}, {10**(parameters[:,0]).max():.2e}]")
    print(f"Frequency: [{np.exp(parameters[:,1]).min():.5e}, {np.exp(parameters[:,1]).max():.5e}]")
    print(f"Frequency derivative: [{np.exp(parameters[:,2]).min():.5e}, {np.exp(parameters[:,2]).max():.5e}]")

if __name__ == "__main__":
    start_time = time.time()
    generate_training_set()
    end_time = time.time()
    print(f"\nGeneration time: {end_time - start_time:.2f} seconds") 