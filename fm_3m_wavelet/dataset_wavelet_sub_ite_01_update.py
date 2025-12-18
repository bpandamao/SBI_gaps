import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
import random
import os

class SpectrogramDatasetH5(Dataset):
    """
    A PyTorch Dataset class to handle loading spectrograms from an HDF5 file.
    The HDF5 file is kept open to avoid I/O overhead on each __getitem__ call.
    It's modified to select a single augmentation per original signal at each call.
    """
    def __init__(self, h5_path, parameters_min, parameters_max, spectrogram_min, spectrogram_max):
        super().__init__()
        self.h5_path = h5_path
        self.parameters_min = torch.tensor(parameters_min, dtype=torch.float32)
        self.parameters_max = torch.tensor(parameters_max, dtype=torch.float32)
        
        # Store normalization parameters
        self.spectrogram_min = spectrogram_min
        self.spectrogram_max = spectrogram_max

        # Add 10% buffer to avoid edge issues
        buffer_spectrogram = 0.1 * (self.spectrogram_max - self.spectrogram_min)
        self.spectrogram_min -= buffer_spectrogram
        self.spectrogram_max += buffer_spectrogram
        
        buffer_params = 0.1 * (self.parameters_max - self.parameters_min)
        self.parameters_min -= buffer_params
        self.parameters_max += buffer_params

        self.file = None
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['parameters'].shape[0] 
            self.num_augmentations = f['spectrograms'].shape[1]

        print(f"Dataset for {os.path.basename(h5_path)} initialized. Using scaling stats: Min={self.spectrogram_min:.4f}, Max={self.spectrogram_max:.4f}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        augmentation_idx = random.randint(0, self.num_augmentations - 1)
        
        params = self.file['parameters'][idx]
        spectrogram = self.file['spectrograms'][idx, augmentation_idx] 
        
        params_tensor = torch.tensor(params, dtype=torch.float32)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        
        # Normalize parameters to [0, 1]
        params_standardized = (params_tensor - self.parameters_min) / (self.parameters_max - self.parameters_min)
        
        # Scale spectrogram to [0, 1] using the stored (training set) stats
        spectrogram_scaled = (spectrogram_tensor - self.spectrogram_min) / (self.spectrogram_max - self.spectrogram_min) 
        
        return spectrogram_scaled.unsqueeze(0), params_standardized

def prepare_data_from_h5(train_h5_path, test_h5_path, parameters_min, parameters_max, batch_size=32, num_workers=4, train_subset_ratio=1, train_skip_interval=None):
    """
    Prepares DataLoader objects for training and testing from separate HDF5 files.
    """
    if train_subset_ratio is not None and train_skip_interval is not None:
        raise ValueError("Cannot use both 'train_subset_ratio' and 'train_skip_interval' simultaneously.")

    # Read normalization stats from the training file
    print(f"Loading normalization statistics from training data: {train_h5_path}")
    with h5py.File(train_h5_path, 'r') as f:
        # Spectrogram stats
        spectrogram_min = f.attrs['spectrogram_min']
        spectrogram_max = f.attrs['spectrogram_max']
    
    print(f"Parameter min: {parameters_min}")
    print(f"Parameter max: {parameters_max}")

    # Create datasets using training set normalization stats
    train_dataset_full = SpectrogramDatasetH5(
        train_h5_path, parameters_min, parameters_max, spectrogram_min, spectrogram_max
    )
    test_dataset = SpectrogramDatasetH5(
        test_h5_path, parameters_min, parameters_max, spectrogram_min, spectrogram_max
    )
    
    if train_subset_ratio is not None:
        num_samples = int(len(train_dataset_full) * train_subset_ratio)
        indices = random.sample(range(len(train_dataset_full)), num_samples)
        train_dataset = Subset(train_dataset_full, indices)
        print(f"Using a random subset of training data (ratio: {train_subset_ratio}). Original size: {len(train_dataset_full)}, Subset size: {len(train_dataset)}")
    elif train_skip_interval is not None:
        if not isinstance(train_skip_interval, int) or train_skip_interval <= 0:
            raise ValueError("'train_skip_interval' must be a positive integer.")
        
        indices = list(range(0, len(train_dataset_full), train_skip_interval))
        train_dataset = Subset(train_dataset_full, indices)
        print(f"Using training data by skipping every {train_skip_interval}-th sample. Original size: {len(train_dataset_full)}, Filtered size: {len(train_dataset)}")
    else:
        train_dataset = train_dataset_full
        print(f"Using full training data. Size: {len(train_dataset)}")
    
    print(f"Validation set size: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader

