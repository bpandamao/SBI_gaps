
import numpy as np
import os
import time
import cupy as xp
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

# Constants
RUN_ID = "ln_30_DAY"
AMPLITUDE_BASE = 5e-21
FREQUENCY_BASE = 2e-3
FREQUENCY_DERIV_BASE = 3e-10
T_MAX = 30 * 24 * 60 * 60  # 30 days in seconds
DT = 5  # 5 seconds sampling
NUM_SAMPLES = 25000
SIGNAL_LENGTH = int(T_MAX / DT)
YRSID_SI = 31558149.763545603

# Fixed parameters for LISA signal generation
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

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        """
        Generate gravitational wave signal for a galactic binary.
        
        Args:
            A: Amplitude
            f: Frequency
            fdot: Frequency derivative
            iota: Inclination angle
            phi0: Initial phase
            psi: Polarization angle
            T: Observation time in years
            dt: Time step in seconds
            
        Returns:
            Complex array representing the gravitational wave signal
        """
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

def generate_training_set():
    """
    Generate training dataset for flow matching models.
    
    Creates a dataset of gravitational wave signals with corresponding parameters
    sampled uniformly in log space. The signals are generated using LISA response
    wrapper and saved to disk.
    """
    output_folder = "training_data"
    os.makedirs(output_folder, exist_ok=True)
    
    # Define true values in linear space
    A_TRUE = AMPLITUDE_BASE
    F_TRUE = FREQUENCY_BASE
    FDOT_TRUE = FREQUENCY_DERIV_BASE

    # Convert true values to log space
    LN_A_TRUE = np.log(A_TRUE)
    LN_F_TRUE = np.log(F_TRUE)
    LN_FDOT_TRUE = np.log(FDOT_TRUE)

    # Define ranges in log space
    LN_A_RANGE = [LN_A_TRUE - 0.1, LN_A_TRUE + 0.1]
    LN_F_RANGE = [LN_F_TRUE - 0.00001, LN_F_TRUE + 0.00001]
    LN_FDOT_RANGE = [LN_FDOT_TRUE - 0.00001, LN_FDOT_TRUE + 0.00001]

    # Generate parameters by sampling uniformly in log space
    ln_a = np.random.uniform(LN_A_RANGE[0], LN_A_RANGE[1], NUM_SAMPLES)
    ln_f = np.random.uniform(LN_F_RANGE[0], LN_F_RANGE[1], NUM_SAMPLES)
    ln_fdot = np.random.uniform(LN_FDOT_RANGE[0], LN_FDOT_RANGE[1], NUM_SAMPLES)

    # Convert the log-space parameters back to linear space for waveform generation
    a = np.exp(ln_a)
    f = np.exp(ln_f)
    fdot = np.exp(ln_fdot)
    
    parameters = np.vstack((ln_a, ln_f, ln_fdot)).T
    
    # Use just enough observation time to generate the 30-day signal
    T_obs = 1.1 * (T_MAX / YRSID_SI)
    
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
        T_obs,
        DT,
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
    
    time_signals = np.empty((NUM_SAMPLES, SIGNAL_LENGTH))
    
    for i in range(NUM_SAMPLES):
        channels = gb_lisa(a[i], f[i], fdot[i], IOTA, PHI0, PSI, LAM, BETA)
        time_signals[i] = np.real(channels[0].get()[:SIGNAL_LENGTH])
        
        if (i+1)%100==0:
            print(f"{i+1}/{NUM_SAMPLES} sets have been generated!")
    
    save_name = os.path.join(output_folder, f"training_set_{RUN_ID}")
    np.savez(f'{save_name}.npz',
             parameters=parameters,
             time_signals=time_signals,
             time_array=np.arange(0, T_MAX, DT))
    
    print(f"\nGenerated training set with {NUM_SAMPLES} samples.")
    print(f"Saved to '{save_name}.npz'")
    

if __name__ == "__main__":
    start_time = time.time()
    generate_training_set()
    end_time = time.time()
    print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
