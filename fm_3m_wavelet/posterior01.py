import torch
import numpy as np
import matplotlib.pyplot as plt
import corner
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils.model_wrapper import ModelWrapper


class WrappedModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        # Explicitly handle the signal parameter
        return self.model(x, t, signal)


def sample_parameters(model, signal, device, x_0=None, num_sample=1000):
    """Generate parameters for a given signal using ODE solver sampling"""
    signal = signal.float()
    if signal.shape[0] != 1:
        raise ValueError("Only one signal can be passed at a time for posterior sampling.")
    
    # Handle different signal dimensions for batch repetition
    # Repeat signal along batch dimension for multiple samples
    if signal.dim() == 4:
        repeated_signal = signal.repeat(num_sample, 1, 1, 1)
    elif signal.dim() == 3:
        repeated_signal = signal.repeat(num_sample, 1, 1)
    else:
        raise ValueError(f"Unsupported signal dimension: {signal.dim()}")

    # Initialize x_0 if not provided
    if x_0 is None:
        x_0 = torch.randn(num_sample, 3, device=device)
    else:
        x_0 = x_0.to(device).float()

    wrapped_model = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Sample using the solver
    samples = solver.sample(
        time_grid=torch.tensor([0.0, 1.0], device=device).float(),
        x_init=x_0,
        method='dopri5',
        step_size=None,
        signal=repeated_signal  # Pass the repeated signal as conditioning
    )
    
    return samples

def analyze_posterior(model, signal, true_params, parameters_min, parameters_max, 
                     n_samples=1000, device="cuda"):
    """Generate and analyze parameter posterior distribution optimized for GPU"""
    # Process in batches to avoid GPU memory issues
    batch_size = 200  # Adjust based on your GPU memory
    all_params = []
    num_batches = n_samples // batch_size + (1 if n_samples % batch_size else 0)
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        batch_params = sample_parameters(
            model, signal, device, 
            num_sample=current_batch_size
        )
        all_params.append(batch_params.cpu())
    
    # Combine batches
    generated_params = torch.cat(all_params, dim=0).numpy()
    
    # Denormalize parameters
    generated_params = generated_params * (parameters_max - parameters_min) + parameters_min
    
    # Convert true_params to numpy without scaling (it's already in the original space)
    true_params = true_params.cpu().numpy().squeeze()
    
    # Create visualization
    labels = ['LN_Amplitude', 'LN_Frequency', 'LN_Frequency dot']
    fig = corner.corner(
        generated_params,
        labels=labels,
        truths=true_params,  # Make sure it's 1D
        truth_color='r',
        show_titles=False,
        title_kwargs={"fontsize": 12},
        quantiles=[0.05, 0.5, 0.95],
        levels=(0.68, 0.95),
        plot_datapoints=True,
        plot_density=True,
    )
    
    # Create subtitle with true values
    true_values_text = "True Values - "
    true_values_text += ", ".join([f"{label}: {value:.2e}" for label, value in zip(labels, true_params)])
    
    plt.suptitle("Parameter Posterior Distribution\n" + true_values_text, 
                 y=1.02, fontsize=10)
    
    return generated_params, fig
