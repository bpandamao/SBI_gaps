import torch
import numpy as np
from tqdm import tqdm
from flow_matching.path import CondOTProbPath
import os
from lisatools.sensitivity import get_sensitivity, A1TDISens
from two_stage_dataset_time import generate_noise
from two_stage_trainer_gaps import create_gaps_mask, standardize_signal, create_gaps_mask_batch

def train_flow_bottleneck(
    flow_model,
    encoder_model, # The pre-trained encoder
    train_loader, 
    test_loader, 
    device, 
    num_epochs=30,
    lr=1e-4, 
    checkpoint_dir=None,
    start_epoch=0,
    signal_length=518400,
    dt=5,
    max_noise_level=1.0,
    curriculum_epochs=0,
    add_gaps=True,
    gap_lambda_value=0.75
):
    """Train the flow matching model using the frozen encoder"""
    
    flow_checkpoint_dir = os.path.join(checkpoint_dir, "flow_matcher")
    os.makedirs(flow_checkpoint_dir, exist_ok=True)
    
    # Setup loss logging file
    loss_log_file = os.path.join(flow_checkpoint_dir, "loss_log.txt")
    if start_epoch == 0:
        try:
            with open(loss_log_file, "w") as f:
                f.write("epoch,train_loss,test_loss,lr,noise_level\n")
        except IOError as e:
            print(f"Warning: Could not write header to loss log file: {e}")

    # Freeze the encoder
    encoder_model.to(device)
    encoder_model.eval()
    for param in encoder_model.parameters():
        param.requires_grad = False
    
    # Optimizer for the flow_model only
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr/10
    )
    
    path = CondOTProbPath()

    train_losses = []
    test_losses = []
    
    if start_epoch > 0:
        checkpoint_path = os.path.join(flow_checkpoint_dir, f'flow_checkpoint_epoch_{start_epoch}.pt')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                flow_model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                train_losses = checkpoint.get('train_losses', [])
                test_losses = checkpoint.get('test_losses', [])
                print(f"Resumed flow matcher training from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading flow checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

    # Pre-calculate PSD
    freqs = np.fft.rfftfreq(signal_length, dt)[1:]
    psd = get_sensitivity(freqs, sens_fn=A1TDISens, return_type="PSD")
    psd_tensor = torch.tensor(psd, device=device)
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        
        # Noise curriculum
        if epoch < curriculum_epochs:
            current_noise_level = max_noise_level * ((epoch + 1) / curriculum_epochs)
        else:
            current_noise_level = max_noise_level

        flow_model.train()
        epoch_loss = 0
        
        for signal, target in train_loader:
            # `signal` is clean, `target` is parameters
            original_signal = signal.to(device).float()
            target = target.to(device).float() # [B, param_dim]
            
            optimizer.zero_grad()
            
            batch_size = original_signal.size(0)
            
            # Create noisy/gapped signal (same as DCAE trainer)
            
            # Vectorized noise generation
            noise_tensor = generate_noise(
                dt, 
                psd_tensor, 
                device=device, 
                noise_level=current_noise_level, 
                N=signal_length, 
                batch_size=batch_size
            ).unsqueeze(1) # [B, 1, N]
            
            noisy_signal_input = original_signal + noise_tensor

            # Vectorized gap creation
            # Default to all ones (no gaps)
            gaps_mask_tensor = torch.ones(batch_size, 1, signal_length, device=device, dtype=torch.float32)
            if add_gaps:
                # create_gaps_mask_batch returns [B, N]
                gaps_mask = create_gaps_mask_batch(
                    batch_size, 
                    signal_length * dt, 
                    dt, 
                    lambda_value=gap_lambda_value, 
                    device=device
                ) # [B, N]
                gaps_mask_tensor = gaps_mask.unsqueeze(1).float() # [B, 1, N]
                noisy_signal_input *= gaps_mask_tensor
            
            # Extract scale info before standardizing, respecting gaps
            # Count non-gap points
            n_non_gap = gaps_mask_tensor.sum(dim=-1, keepdim=True)  # [B, 1, 1]
            
            # Calculate mean only on non-gap points
            mean = (noisy_signal_input * gaps_mask_tensor).sum(dim=-1, keepdim=True) / n_non_gap
            
            # Calculate variance only on non-gap points
            # Unbiased variance divides by (N-1)
            variance = ((noisy_signal_input - mean).pow(2) * gaps_mask_tensor).sum(dim=-1, keepdim=True) / (n_non_gap - 1)
            
            # Calculate std directly without clamps
            signal_std = torch.sqrt(variance)  # [B, 1, 1]
            log_signal_std = torch.log(signal_std).squeeze(-1)  # [B, 1]
            
            # Standardize the input signal after getting std
            noisy_signal_input = standardize_signal(noisy_signal_input)
            
            # Get bottleneck from frozen encoder
            with torch.no_grad():
                bottleneck = encoder_model(noisy_signal_input.float())  # [B, 256]
            
            # Combine bottleneck and scale info
            combined_conditioning = torch.cat([bottleneck, log_signal_std.float()], dim=1)  # [B, 257]
            
            # Standard flow matching loss (using combined condition)
            t = torch.rand(target.shape[0], device=device)
            noise = torch.randn_like(target).to(device)
            path_sample = path.sample(t=t, x_0=noise, x_1=target)
            x_t = path_sample.x_t.to(device)
            u_t = path_sample.dx_t.to(device)
            
            # Pass combined_conditioning as the conditioning signal
            pred = flow_model(x_t, t, combined_conditioning) 
            loss = torch.pow(pred - u_t, 2).mean()
            
            loss.backward()  # Gradients only flow through flow_model
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        flow_model.eval()
        test_loss = 0
        with torch.no_grad():
            for signal, target in test_loader:
                
                original_signal = signal.to(device).float()
                target = target.to(device).float()
                batch_size = original_signal.size(0)

                # Vectorized noise generation
                noise_tensor = generate_noise(
                    dt, 
                    psd_tensor, 
                    device=device, 
                    noise_level=current_noise_level, 
                    N=signal_length, 
                    batch_size=batch_size
                ).unsqueeze(1) # [B, 1, N]
                
                noisy_signal_input = original_signal + noise_tensor

                # Vectorized gap creation
                gaps_mask_tensor = torch.ones(batch_size, 1, signal_length, device=device, dtype=torch.float32)
                if add_gaps:
                    gaps_mask = create_gaps_mask_batch(
                        batch_size, 
                        signal_length * dt, 
                        dt, 
                        lambda_value=gap_lambda_value, 
                        device=device
                    ) # [B, N]
                    gaps_mask_tensor = gaps_mask.unsqueeze(1).float() # [B, 1, N]
                    noisy_signal_input *= gaps_mask_tensor

                # Extract scale info before standardizing, respecting gaps
                # Count non-gap points
                n_non_gap = gaps_mask_tensor.sum(dim=-1, keepdim=True)  # [B, 1, 1]
                mean = (noisy_signal_input * gaps_mask_tensor).sum(dim=-1, keepdim=True) / n_non_gap
                variance = ((noisy_signal_input - mean).pow(2) * gaps_mask_tensor).sum(dim=-1, keepdim=True) / (n_non_gap - 1)
                
                # Calculate std directly without clamps
                signal_std = torch.sqrt(variance)  # [B, 1, 1]
                log_signal_std = torch.log(signal_std).squeeze(-1)  # [B, 1]
                
                # Standardize the input signal after getting std
                noisy_signal_input = standardize_signal(noisy_signal_input)
                
                bottleneck = encoder_model(noisy_signal_input.float())  # [B, 256]
                
                # Combine bottleneck and scale info
                combined_conditioning = torch.cat([bottleneck, log_signal_std.float()], dim=1)  # [B, 257]

                t = torch.rand(target.shape[0], device=device)
                noise = torch.randn_like(target).to(device)
                path_sample = path.sample(t=t, x_0=noise, x_1=target)
                x_t = path_sample.x_t.to(device)
                u_t = path_sample.dx_t.to(device)
                
                pred = flow_model(x_t, t, combined_conditioning)
                loss = torch.pow(pred - u_t, 2).mean()
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'\n--- Flow Matcher Epoch [{epoch+1}/{num_epochs}] ---')
            print(f'Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
            print(f'Noise Level: {current_noise_level:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Log to text file
        try:
            with open(loss_log_file, "a") as f:
                f.write(f"{epoch + 1},{avg_train_loss:.8f},{avg_test_loss:.8f},{scheduler.get_last_lr()[0]:.8f},{current_noise_level:.4f}\n")
        except IOError as e:
            print(f"Warning: Could not append to loss log file: {e}")

        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(flow_checkpoint_dir, f'flow_checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1, 
                'model_state_dict': flow_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses, 
                'test_losses': test_losses,
            }, checkpoint_path)

    return flow_model, train_losses, test_losses