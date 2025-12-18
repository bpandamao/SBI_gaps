"""
Complete Pipeline Demo for Flow Matching with Wavelet Spectrograms

This script provides a complete, self-contained demo that:
1. Generates training data (clean signals)
2. Augments data with noise and gaps
3. Trains flow matching model with symmetric or asymmetric kernels with dilation

The script handles file paths and configurations automatically.

Usage:
    # Train with symmetric kernels
    python demo_complete_pipeline.py --kernel_type symmetric --num_samples 100 --num_epochs 50

    # Train with asymmetric kernels  
    python demo_complete_pipeline.py --kernel_type asymmetric --num_samples 100 --num_epochs 50
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from dataset_wavelet_sub_ite_01_update import prepare_data_from_h5
from trainer_spectrogramV1 import train_flow_matching_spectrogram
from model_utils import print_model_structure

# Import flow matchers
from flow_matcher_time import ContinuousFlowMatcherTime as SymmetricFlowMatcher
from flow_matcher_time_asy import ContinuousFlowMatcherTime as AsymmetricFlowMatcher


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Complete Flow Matching Pipeline Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--kernel_type', type=str, default='symmetric',
                        choices=['symmetric', 'asymmetric'],
                        help='Type of kernel: symmetric (3x3) or asymmetric (3x9)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of training samples to generate')
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help='Number of augmentations per signal')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--signal_embedding_dim', type=int, default=512,
                        help='Signal embedding dimension')
    parser.add_argument('--output_dir', type=str, default='demo_outputs',
                        help='Output directory for all results')
    parser.add_argument('--use_existing_data', action='store_true',
                        help='Use existing data files if available')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0,
                        help='Fraction of training data to use (for faster testing)')
    return parser.parse_args()


class DataGenerator:
    """Wrapper for generating training data"""
    
    def __init__(self, num_samples, output_dir):
        self.num_samples = num_samples
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self):
        """Generate clean training signals"""
        print("\n" + "="*70)
        print("STEP 1: Generating Training Data")
        print("="*70)
        
        # Import here to avoid issues if module not available
        try:
            from training_data_generator_time_ln_likevb_smaller import (
                generate_training_set, NUM_SAMPLES, RUN_ID
            )
        except ImportError:
            print("ERROR: Could not import training_data_generator_time_ln_likevb_smaller")
            print("Please ensure the module is available.")
            return None
        
        # Modify constants temporarily
        import training_data_generator_time_ln_likevb_smaller as gen_module
        original_num_samples = gen_module.NUM_SAMPLES
        original_run_id = gen_module.RUN_ID
        
        gen_module.NUM_SAMPLES = self.num_samples
        gen_module.RUN_ID = f"demo_{self.num_samples}samples"
        
        try:
            generate_training_set()
            
            # Find generated file
            training_data_dir = "training_data"
            npz_files = list(Path(training_data_dir).glob("fullsignal_*.npz"))
            
            if npz_files:
                training_file = str(sorted(npz_files, key=os.path.getmtime)[-1])
                print(f"✓ Generated training data: {training_file}")
                return training_file
            else:
                print("✗ Training data file not found")
                return None
        finally:
            # Restore original values
            gen_module.NUM_SAMPLES = original_num_samples
            gen_module.RUN_ID = original_run_id


class DataAugmenter:
    """Wrapper for augmenting data with noise and gaps"""
    
    def __init__(self, num_augmentations, output_dir):
        self.num_augmentations = num_augmentations
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def augment(self, input_npz_file):
        """Augment training data with noise and gaps"""
        print("\n" + "="*70)
        print("STEP 2: Augmenting Data with Noise and Gaps")
        print("="*70)
        
        try:
            import augment_and_wavelet_gaps_ite_01_VB as aug_module
        except ImportError:
            print("ERROR: Could not import augment_and_wavelet_gaps_ite_01_VB")
            return None
        
        # Create output filename
        input_name = Path(input_npz_file).stem
        output_h5_file = os.path.join(self.output_dir, f"{input_name}_augmented.h5")
        
        # Modify module constants
        original_input = aug_module.INPUT_DATA_FILE
        original_output = aug_module.OUTPUT_H5_FILE
        original_num_aug = aug_module.NOISE_AUGMENTATIONS
        
        aug_module.INPUT_DATA_FILE = input_npz_file
        aug_module.OUTPUT_H5_FILE = output_h5_file
        aug_module.NOISE_AUGMENTATIONS = self.num_augmentations
        
        try:
            print(f"Input: {input_npz_file}")
            print(f"Output: {output_h5_file}")
            print(f"Augmentations per signal: {self.num_augmentations}")
            
            aug_module.augment_and_transform_parallel()
            
            if os.path.exists(output_h5_file):
                print(f"✓ Augmented data saved: {output_h5_file}")
                return output_h5_file
            else:
                print("✗ Augmented data file not found")
                return None
        finally:
            # Restore original values
            aug_module.INPUT_DATA_FILE = original_input
            aug_module.OUTPUT_H5_FILE = original_output
            aug_module.NOISE_AUGMENTATIONS = original_num_aug


class FlowMatchingTrainer:
    """Wrapper for training flow matching models"""
    
    def __init__(self, kernel_type, output_dir, batch_size, num_epochs, 
                 signal_embedding_dim, train_subset_ratio=1.0):
        self.kernel_type = kernel_type
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.signal_embedding_dim = signal_embedding_dim
        self.train_subset_ratio = train_subset_ratio
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self, train_h5_file, test_h5_file=None):
        """Train flow matching model"""
        print("\n" + "="*70)
        print(f"STEP 3: Training Flow Matching Model ({self.kernel_type.upper()} kernel)")
        print("="*70)
        
        if test_h5_file is None:
            test_h5_file = train_h5_file
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Parameter ranges
        AMPLITUDE_BASE = 1.5e-21
        FREQUENCY_BASE = 2e-3
        FREQUENCY_DERIV_BASE = 1e-10
        
        params_min = np.array([
            np.log10(AMPLITUDE_BASE) - 0.1,
            np.log(FREQUENCY_BASE) - 0.001,
            np.log(FREQUENCY_DERIV_BASE) - 0.001
        ])
        params_max = np.array([
            np.log10(AMPLITUDE_BASE) + 0.1,
            np.log(FREQUENCY_BASE) + 0.001,
            np.log(FREQUENCY_DERIV_BASE) + 0.001
        ])
        
        # Load data
        print("Loading data...")
        train_loader, test_loader = prepare_data_from_h5(
            train_h5_path=train_h5_file,
            test_h5_path=test_h5_file,
            parameters_min=params_min,
            parameters_max=params_max,
            batch_size=self.batch_size,
            num_workers=4,
            train_subset_ratio=self.train_subset_ratio
        )
        
        # Load spectrogram stats
        with h5py.File(train_h5_file, 'r') as f:
            spectrogram_min = f.attrs['spectrogram_min']
            spectrogram_max = f.attrs['spectrogram_max']
        
        # Add buffer
        buffer = 0.1 * (spectrogram_max - spectrogram_min)
        spectrogram_min -= buffer
        spectrogram_max += buffer
        
        print(f"Spectrogram range: [{spectrogram_min:.4f}, {spectrogram_max:.4f}]")
        
        # Create model
        SIGNAL_LENGTH = 1572864
        
        if self.kernel_type == 'symmetric':
            model = SymmetricFlowMatcher(
                param_dim=3,
                signal_embedding_dim=self.signal_embedding_dim,
                signal_input_dim=SIGNAL_LENGTH
            ).to(device)
            print("Model: Symmetric kernels (3x3) with dilation")
        else:
            model = AsymmetricFlowMatcher(
                param_dim=3,
                signal_embedding_dim=self.signal_embedding_dim,
                signal_input_dim=SIGNAL_LENGTH
            ).to(device)
            print("Model: Asymmetric kernels (3x9) with dilation")
        
        # Print model structure
        model_structure_file = os.path.join(self.output_dir, "model_structure.txt")
        print_model_structure(model, filename=model_structure_file)
        
        # Save model info
        np.savez(
            os.path.join(self.output_dir, "model_info.npz"),
            params_min=params_min,
            params_max=params_max,
            spectrogram_min=spectrogram_min,
            spectrogram_max=spectrogram_max,
            kernel_type=self.kernel_type,
            signal_embedding_dim=self.signal_embedding_dim
        )
        
        # Train
        print(f"\nTraining for {self.num_epochs} epochs...")
        start_time = time.time()
        
        model, train_losses, test_losses = train_flow_matching_spectrogram(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=self.num_epochs,
            lr_stage1=1e-3,
            lr_stage2=1e-4,
            stage1_epochs=int(0.7 * self.num_epochs),
            stage2_epochs=int(0.3 * self.num_epochs),
            checkpoint_dir=None,
            start_epoch=0
        )
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
        
        # Save model
        model_file = os.path.join(self.output_dir, "flow_matcher.pt")
        torch.save(model.state_dict(), model_file)
        print(f"✓ Model saved: {model_file}")
        
        # Save losses
        loss_file = os.path.join(self.output_dir, "losses.npz")
        np.savez(loss_file, train_losses=train_losses, test_losses=test_losses)
        
        # Plot losses
        self._plot_losses(train_losses, test_losses)
        
        return model, train_losses, test_losses
    
    def _plot_losses(self, train_losses, test_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
        plt.plot(epochs, test_losses, 'r-', label='Validation Loss', alpha=0.7)
        plt.title(f'Training and Validation Loss ({self.kernel_type.upper()} kernel)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_plot_file = os.path.join(self.output_dir, "loss_plot.png")
        plt.savefig(loss_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss plot saved: {loss_plot_file}")


def main():
    """Main demo function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("FLOW MATCHING COMPLETE PIPELINE DEMO")
    print("="*70)
    print(f"Kernel Type:        {args.kernel_type}")
    print(f"Num Samples:        {args.num_samples}")
    print(f"Num Augmentations:  {args.num_augmentations}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Num Epochs:         {args.num_epochs}")
    print(f"Output Directory:   {args.output_dir}")
    print("="*70)
    
    try:
        # Step 1: Generate training data
        data_gen = DataGenerator(args.num_samples, 
                                os.path.join(args.output_dir, "training_data"))
        
        if args.use_existing_data:
            # Try to find existing training data
            training_data_dir = "training_data"
            npz_files = list(Path(training_data_dir).glob("fullsignal_*.npz"))
            if npz_files:
                training_file = str(sorted(npz_files, key=os.path.getmtime)[-1])
                print(f"\nUsing existing training data: {training_file}")
            else:
                training_file = data_gen.generate()
        else:
            training_file = data_gen.generate()
        
        if training_file is None:
            print("\n✗ Failed to generate/find training data")
            return 1
        
        # Step 2: Augment data
        aug_dir = os.path.join(args.output_dir, "augmented_data")
        data_aug = DataAugmenter(args.num_augmentations, aug_dir)
        
        if args.use_existing_data:
            # Try to find existing augmented data
            h5_files = list(Path(aug_dir).glob("*_augmented.h5"))
            if not h5_files:
                h5_files = list(Path("training_data").glob("au_sp_*.h5"))
            
            if h5_files:
                augmented_file = str(sorted(h5_files, key=os.path.getmtime)[-1])
                print(f"\nUsing existing augmented data: {augmented_file}")
            else:
                augmented_file = data_aug.augment(training_file)
        else:
            augmented_file = data_aug.augment(training_file)
        
        if augmented_file is None:
            print("\n✗ Failed to generate/find augmented data")
            return 1
        
        # Step 3: Train model
        model_dir = os.path.join(args.output_dir, f"model_{args.kernel_type}")
        trainer = FlowMatchingTrainer(
            kernel_type=args.kernel_type,
            output_dir=model_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            signal_embedding_dim=args.signal_embedding_dim,
            train_subset_ratio=args.train_subset_ratio
        )
        
        model, train_losses, test_losses = trainer.train(augmented_file)
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Final Training Loss: {train_losses[-1]:.6f}")
        print(f"Final Validation Loss: {test_losses[-1]:.6f}")
        print(f"\nAll outputs saved to: {args.output_dir}")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

