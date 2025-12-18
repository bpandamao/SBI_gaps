import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from lisatools.sensitivity import get_sensitivity, A1TDISens
from two_stage_dataset_time import generate_noise
from two_stage_trainer_gaps import create_gaps_mask, standardize_signal, create_gaps_mask_batch

def train_dcae(
    model, 
    train_loader, 
    test_loader, 
    device, 
    num_epochs=150,
    lr=1e-4, 
    checkpoint_dir=None,
    start_epoch=0,
    signal_length=518400,
    dt=5,
    max_noise_level=1.0,
    curriculum_epochs=100,
    add_gaps=True,
    gap_lambda_value=0.75,
    accumulation_steps=1  # Gradient accumulation
):
    """Train the Denoising Convolutional Autoencoder (DCAE)"""
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    
    # Use MSELoss for reconstruction
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f'dcae_checkpoint_epoch_{start_epoch}.pt')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                train_losses = checkpoint.get('train_losses', [])
                test_losses = checkpoint.get('test_losses', [])
                print(f"Resumed training from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

    # Pre-calculate PSD
    freqs = np.fft.rfftfreq(signal_length, dt)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor = torch.tensor(psd, device=device)
    
    print(f"Starting DCAE training with accumulation_steps = {accumulation_steps}")
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        
        # Noise curriculum
        if epoch < curriculum_epochs:
            current_noise_level = max_noise_level * ((epoch + 1) / curriculum_epochs)
        else:
            current_noise_level = max_noise_level

        model.train()
        epoch_loss = 0
        
        optimizer.zero_grad() # <-- Moved outside the loop
        
        for i, (signal, target_params) in enumerate(train_loader):
            
            original_signal = signal.to(device).float()
            
            batch_size = original_signal.size(0)
            
            # Vectorized noise generation
            # 1. Generate noise (batched)
            noise_tensor = generate_noise(
                dt, 
                psd_tensor, 
                device=device, 
                noise_level=current_noise_level, 
                N=signal_length, 
                batch_size=batch_size
            ).unsqueeze(1) # Add channel dim: [B, 1, N]
            
            noisy_signal_input = original_signal + noise_tensor

            # Vectorized gap creation
            # 2. Apply gaps (batched)
            if add_gaps:
                gaps_mask_tensor = create_gaps_mask_batch(
                    batch_size, 
                    signal_length * dt, 
                    dt, 
                    lambda_value=gap_lambda_value, 
                    device=device
                ).unsqueeze(1) # Add channel dim: [B, 1, N]
                noisy_signal_input *= gaps_mask_tensor
            
            # 3. Standardize the *input* signal
            noisy_signal_input = standardize_signal(noisy_signal_input)
            
            # 4. Standardize the *target* signal (as discussed)
            standardized_original_signal = standardize_signal(original_signal)
            
            # Forward pass
            reconstructed_signal, _ = model(noisy_signal_input.float())
            
            # Calculate loss
            loss = criterion(reconstructed_signal, standardized_original_signal.float())
            
            # Normalize loss for accumulation
            loss = loss / accumulation_steps
            
            loss.backward()
            
            epoch_loss += loss.item() * accumulation_steps  # Scale back up for logging
            
            # Step optimizer only after N steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        # Handle the last few batches if they don't divide evenly
        if (len(train_loader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loop (no accumulation needed)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for signal, target_params in test_loader:
                
                original_signal = signal.to(device).float()
                batch_size = original_signal.size(0)

                # Vectorized noise generation
                noise_tensor = generate_noise(
                    dt, 
                    psd_tensor, 
                    device=device, 
                    noise_level=current_noise_level, 
                    N=signal_length, 
                    batch_size=batch_size
                ).unsqueeze(1) # Add channel dim: [B, 1, N]
                
                noisy_signal_input = original_signal + noise_tensor

                # Vectorized gap creation
                if add_gaps:
                    gaps_mask_tensor = create_gaps_mask_batch(
                        batch_size, 
                        signal_length * dt, 
                        dt, 
                        lambda_value=gap_lambda_value, 
                        device=device
                    ).unsqueeze(1) # Add channel dim: [B, 1, N]
                    noisy_signal_input *= gaps_mask_tensor
                
                noisy_signal_input = standardize_signal(noisy_signal_input)
                
                standardized_original_signal = standardize_signal(original_signal)
                
                reconstructed_signal, _ = model(noisy_signal_input.float())
                
                loss = criterion(reconstructed_signal, standardized_original_signal.float())
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'\n--- DCAE Epoch [{epoch+1}/{num_epochs}] ---')
            print(f'Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
            print(f'Noise Level: {current_noise_level:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'dcae_checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses, 
                'test_losses': test_losses,
            }, checkpoint_path)

    return model, train_losses, test_losses