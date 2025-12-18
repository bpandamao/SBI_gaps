import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
# from lisatools.sensitivity import get_sensitivity, A1TDISens

class TimeSignalDataset(Dataset):
    def __init__(self, parameters, time_signals, device=None):
        self.parameters = parameters
        self.time_signals = time_signals
        self.device = device

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        params = self.parameters[idx]
        signal = self.time_signals[idx]
        
        # Signal is already in the expected format [1, signal_dim]
        if self.device is not None:
            return torch.tensor(signal, dtype=torch.float32, device=self.device), torch.tensor(params, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(signal, dtype=torch.float32), torch.tensor(params, dtype=torch.float32)

def generate_noise(dt, psd_tensor, noise_level=1, device=None,N=518400,batch_size=1):
    """
    Generate noise in time domain based on given PSD using PyTorch FFT on GPU
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N_t = int(2**(torch.ceil(torch.log2(torch.tensor(N, device=device)))))
    freqs = torch.fft.rfftfreq(N, dt, device=device)[1:]  # Exclude DC component
 
    # # Convert PSD to tensor and move to GPU
    # psd_tensor = torch.tensor(psd, device=device)  
    # Calculate variance in frequency domain
    variance_noise_f = (N_t * psd_tensor / (4 * dt))
    
    # Generate complex Gaussian noise
    # --- MODIFIED to support batch_size ---
    noise_f_real = torch.randn(batch_size, len(freqs), device=device)
    noise_f_imag = torch.randn(batch_size, len(freqs), device=device)
    noise_f = torch.sqrt(variance_noise_f) * (noise_f_real + 1j * noise_f_imag)
    
    # Add DC component (zero)
    noise_f = torch.cat((torch.zeros(batch_size, 1, device=device), noise_f), dim=1)
    
    # Transform to time domain
    noise_t = torch.fft.irfft(noise_f, n=N_t, dim=1)[:, :N] # [batch_size, N]
    
    output_noise_t = noise_t * noise_level  # Scale by sqrt(2*dt)
    return output_noise_t

def prepare_data(parameters, time_signals, batch_size=32, train_split=0.8, device=None):
    """
    Prepare data for training by normalizing parameters.
    Signal processing will be done in the trainer.
    
    Args:
        parameters: Array of parameters [n_samples, n_params]
        time_signals: Array of time signals [n_samples, 1, signal_length]
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        device: Device to store tensors on (e.g., 'cuda' or 'cpu')
    
    Returns:
        train_loader, test_loader, parameters_mean, parameters_std
    """
    # Parameter normalization
    parameters_mean = np.mean(parameters, axis=0)
    parameters_std = np.std(parameters, axis=0)
    parameters_standardized = ((parameters - parameters_mean) / parameters_std)
    
    # Create dataset with raw signals (no processing)
    dataset = TimeSignalDataset(parameters_standardized, time_signals, device=device)
    
    # Split into train and test sets
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           num_workers=0, pin_memory=True, persistent_workers=False)
    
    return train_loader, test_loader, parameters_mean, parameters_std 