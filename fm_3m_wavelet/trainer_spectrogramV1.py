import torch
from tqdm import tqdm
from flow_matching.path import CondOTProbPath
import os

def train_flow_matching_spectrogram(
    model, 
    train_loader, 
    test_loader, 
    device, 
    num_epochs=500,
    lr_stage1=1e-3, 
    lr_stage2=1e-4, 
    stage1_epochs=350,
    stage2_epochs=150,
    checkpoint_dir=None,
    start_epoch=0,
    new_lr_multiplier_stage1=1,
    new_lr_multiplier_stage2=1,
    eta_min_ratio=10,
):
    """
    Train the flow matching model using pre-computed spectrograms with a 
    two-stage learning rate schedule.
    """
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Separate parameters for different learning rates for Stage 1
    existing_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'signal_embedding' in name:
            new_params.append(param)
        else:
            existing_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': existing_params, 'lr': lr_stage1},
        {'params': new_params, 'lr': lr_stage1 * new_lr_multiplier_stage1}
    ])
    
    # Scheduler for Stage 1, T_max is now the full length of the stage
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage1_epochs, eta_min=lr_stage1 / 10
    )
    
    path = CondOTProbPath()
    train_losses, test_losses = [], []
    
    # Load from checkpoint if available
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{start_epoch}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Epochs"):
        
        # --- Training Loop ---
        model.train()
        epoch_loss = 0
        for spectrogram, target in train_loader:
            optimizer.zero_grad()
            
            # Data comes pre-processed from the DataLoader
            spectrogram = spectrogram.to(device)
            target = target.to(device)
            
            t = torch.rand(target.shape[0], device=device)
            noise_param = torch.randn_like(target)
            path_sample = path.sample(t=t, x_0=noise_param, x_1=target)
            x_t, u_t = path_sample.x_t, path_sample.dx_t
            # print(spectrogram.shape) 
            
            pred = model(x_t, t, spectrogram)
            loss = torch.pow(pred - u_t, 2).mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation Loop ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for spectrogram, target in test_loader:
                spectrogram = spectrogram.to(device)
                target = target.to(device)
                
                t = torch.rand(target.shape[0], device=device)
                noise_param = torch.randn_like(target)
                path_sample = path.sample(t=t, x_0=noise_param, x_1=target)
                x_t, u_t = path_sample.x_t, path_sample.dx_t
                
                pred = model(x_t, t, spectrogram)
                loss = torch.pow(pred - u_t, 2).mean()
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # --- Two-Stage Learning Rate and Scheduler Logic ---
        if epoch < stage1_epochs:
            # Step the scheduler for stage 1
            scheduler.step()
        elif epoch == stage1_epochs:
            # Transition to Stage 2
            print(f"\n--- Switching to Stage 2 Learning Rate at Epoch {epoch + 1} ---")
            optimizer = torch.optim.Adam([
                {'params': existing_params, 'lr': lr_stage2},
                {'params': new_params, 'lr': lr_stage2 * new_lr_multiplier_stage2}
            ])
            # Create a new scheduler for Stage 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=lr_stage2 / 5
            )
        else:
            # We are in Stage 2, continue stepping the new scheduler
            scheduler.step()
        
        # # --- Checkpointing and Logging ---
        # if checkpoint_dir and (epoch + 1) % 1000 == 0:
        #     torch.save({
        #         'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_losses': train_losses, 'test_losses': test_losses,
        #     }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {current_lr:.2e}')

    return model, train_losses, test_losses
