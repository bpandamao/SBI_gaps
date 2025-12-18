import torch
import numpy as np
from tqdm import tqdm
from flow_matching.path import CondOTProbPath
import os
from lisatools.sensitivity import get_sensitivity, A1TDISens
from dataset_time import generate_noise

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

def standardize_signal(signal):
    """
    Standardize a signal by subtracting mean and dividing by std
    """
    mean = torch.mean(signal, dim=-1, keepdim=True)
    std = torch.std(signal, dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero for silent signals (gaps)
    return (signal - mean) / (std)

def train_flow_matching(
    model, 
    train_loader, 
    test_loader, 
    device, 
    num_epochs=500,
    curriculum_epochs=400,
    max_noise_level=1.0,
    lr_stage1=1e-3, 
    lr_stage2=1e-4, 
    early_stop=False, 
    stage1_epochs=350, 
    stage2_epochs=150,
    checkpoint_dir=None,
    start_epoch=0,
    warmup_epochs=10,
    new_lr_multiplier_stage1=1,
    new_lr_multiplier_stage2=1,
    eta_min_ratio=10,
    eta_min_ratio2=5,
    num_workers=4,
    add_gaps=False, # Control whether to add gaps
    gap_lambda_value=0.75 # Lambda for gap creation
):
    """Train the flow matching model and evaluate on test set"""
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    existing_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'conv_layers' in name or 'final_proj' in name:
            new_params.append(param)
        else:
            existing_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': existing_params, 'lr': lr_stage1},
        {'params': new_params, 'lr': lr_stage1 * new_lr_multiplier_stage1}
    ])
    
    current_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage1_epochs, eta_min=lr_stage1/eta_min_ratio
    )
    
    if early_stop:
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_model_state = None
        min_epochs = 100
        min_delta = 0.001
    
    path = CondOTProbPath()

    duration = 516096*5
    delta_t = 5
    
    train_losses = []
    test_losses = []
    
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{start_epoch}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])

    signal_length = int(duration / delta_t)
    freqs = np.fft.rfftfreq(signal_length, delta_t)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor = torch.tensor(psd, device=device)
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        if epoch < curriculum_epochs:
            current_noise_level = max_noise_level * ((epoch + 1) / curriculum_epochs)
        else:
            current_noise_level = max_noise_level

        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                if param_group['params'] == new_params:
                    param_group['lr'] = lr_stage1 * new_lr_multiplier_stage1 * warmup_factor
                else:
                    param_group['lr'] = lr_stage1
        
        model.train()
        epoch_loss = 0
        
        for signal, target in train_loader:
            if signal.dim() != 3 or signal.size(1) != 1:
                raise ValueError(f"Signal shape mismatch... got shape {signal.shape}")
            
            optimizer.zero_grad()
            
            signal = signal.to(device).float()
            target = target.to(device).float()
            
            batch_size = signal.size(0)
            noise_tensor = torch.zeros_like(signal, device=device, dtype=torch.float32)
            
            for i in range(batch_size):
                noise = generate_noise(delta_t, psd_tensor, device=device, noise_level=current_noise_level)
                noise_tensor[i, 0, :] = noise[:signal_length]
            
            noisysignal = signal + noise_tensor

            # Apply gaps if enabled
            if add_gaps:
                for i in range(batch_size):
                    gaps_mask = create_gaps_mask(duration, delta_t, lambda_value=gap_lambda_value)
                    gaps_mask_tensor = torch.from_numpy(gaps_mask).to(device)
                    noisysignal[i, 0, :] *= gaps_mask_tensor
            
            noisysignal = standardize_signal(noisysignal)
            
            t = torch.rand(target.shape[0], device=device)
            noise = torch.randn_like(target).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=target)
            x_t = path_sample.x_t.to(device)
            u_t = path_sample.dx_t.to(device)
            
            pred = model(x_t, t, noisysignal)
            loss = torch.pow(pred - u_t, 2).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for signal, target in test_loader:
                signal = signal.to(device).float()
                target = target.to(device).float()
                batch_size = signal.size(0)
                noise_tensor = torch.zeros_like(signal, device=device, dtype=torch.float32)
                
                for i in range(batch_size):
                    noise = generate_noise(delta_t, psd_tensor, device=device, noise_level=current_noise_level)
                    noise_tensor[i, 0, :] = noise[:signal_length]
                
                noisysignal = signal + noise_tensor

                # Apply gaps if enabled
                if add_gaps:
                    for i in range(batch_size):
                        gaps_mask = create_gaps_mask(duration, delta_t, lambda_value=gap_lambda_value)
                        gaps_mask_tensor = torch.from_numpy(gaps_mask).to(device)
                        noisysignal[i, 0, :] *= gaps_mask_tensor

                noisysignal = standardize_signal(noisysignal)
                
                t = torch.rand(target.shape[0], device=device)
                noise = torch.randn_like(target).to(device)
                path_sample = path.sample(t=t, x_0=noise, x_1=target)
                x_t = path_sample.x_t.to(device)
                u_t = path_sample.dx_t.to(device)
                
                pred = model(x_t, t, noisysignal)
                loss = torch.pow(pred - u_t, 2).mean()
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        if checkpoint_dir and (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': current_scheduler.state_dict(),
                'train_losses': train_losses, 'test_losses': test_losses,
            }, checkpoint_path)
        
        if early_stop and epoch >= min_epochs:
            if avg_test_loss < best_loss - min_delta:
                best_loss = avg_test_loss
                patience_counter = 0
                best_model_state = {key: val.cpu() for key, val in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                model.load_state_dict(best_model_state)
                break
        
        if epoch < stage1_epochs:
            current_scheduler.step()
        elif epoch == stage1_epochs:
            optimizer = torch.optim.Adam([
                {'params': existing_params, 'lr': lr_stage2},
                {'params': new_params, 'lr': lr_stage2 * new_lr_multiplier_stage2}
            ])
            current_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=lr_stage2/eta_min_ratio2
            )
        else:
            current_scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
            print(f'Current Noise Level: {current_noise_level:.4f}')
            if add_gaps:
                print(f'Gaps enabled with lambda: {gap_lambda_value}')
            if early_stop and epoch >= min_epochs:
                print(f'Patience: {patience_counter}/{patience}, Best Loss: {best_loss:.4f}')

    return model, train_losses, test_losses