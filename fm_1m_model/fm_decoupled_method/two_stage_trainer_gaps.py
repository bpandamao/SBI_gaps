import torch
import numpy as np
from tqdm import tqdm
from flow_matching.path import CondOTProbPath
import os
from lisatools.sensitivity import get_sensitivity, A1TDISens
from two_stage_dataset_time import generate_noise

# --- NEW: Function to create data gaps ---
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

# --- NEW: Batch-based gap creation ---
def create_gaps_mask_batch(batch_size, duration, dt, lambda_value=0.75, device='cuda'):
    """
    Create a batch of masks for gaps in the data using PyTorch.
    
    Args:
        batch_size (int): Number of masks to generate.
        duration (float): Total duration of the signal in seconds.
        dt (float): The time step (sampling interval).
        lambda_value (float): Parameter for the exponential distribution.
        device (str): The device to create tensors on.
        
    Returns:
        torch.Tensor: A binary mask tensor of shape [batch_size, total_points].
    """
    total_points = int(duration / dt)
    mask = torch.ones(batch_size, total_points, device=device, dtype=torch.float32)
    
    current_points = torch.zeros(batch_size, device=device, dtype=torch.long)
    
    # We need to loop, but the operations inside are batched
    # This is much faster than N separate CPU loops
    while torch.any(current_points < total_points):
        # 1. Generate data segment lengths (Exponential)
        # scale = 1 / lambda_value
        # Using torch.rand for exponential distribution
        data_length_days = -torch.log(torch.rand(batch_size, device=device) + 1e-20) / lambda_value
        data_length_points = (data_length_days * 24 * 3600 / dt).long()
        
        current_points += data_length_points
        
        # Check which signals are already finished
        finished_mask = current_points >= total_points
        if torch.all(finished_mask):
            break # All signals are done
        
        # 2. Generate gap lengths (Uniform 3.5 to 12 hours)
        gap_duration_hours = torch.rand(batch_size, device=device) * (12.0 - 3.5) + 3.5
        gap_duration_points = (gap_duration_hours * 3600 / dt).long()
        
        # Find start and end of gap for each signal in the batch
        gap_start_points = current_points.clone()
        gap_end_points = torch.min(gap_start_points + gap_duration_points, torch.tensor(total_points, device=device))
        
        # Create indices to set gaps to 0
        # A simple (but not fully parallel) loop is still *vastly* faster
        # than the numpy loop because we are not transferring CPU->GPU.
        for i in range(batch_size):
            if not finished_mask[i]:
                # Clamp start index just in case
                start_idx = torch.clamp(gap_start_points[i], 0, total_points)
                end_idx = torch.clamp(gap_end_points[i], 0, total_points)
                if start_idx < end_idx:
                    mask[i, start_idx:end_idx] = 0
                        
        current_points += gap_duration_points

    return mask

def standardize_signal(signal):
    """
    Standardize a signal by subtracting mean and dividing by std
    """
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero for silent signals (gaps)
    return (signal - mean) / (std)