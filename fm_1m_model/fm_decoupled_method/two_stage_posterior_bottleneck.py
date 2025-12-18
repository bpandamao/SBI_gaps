import torch
import numpy as np
import matplotlib.pyplot as plt
import corner
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils.model_wrapper import ModelWrapper


class WrappedModel(ModelWrapper):
    """
    Wrapper for the ODESolver, adapted for the bottleneck model.
    """
    def __init__(self, model):
        super().__init__(model)

    # --- FIX ---
    # The ODESolver passes the conditioning vector as the 'signal' keyword argument.
    # We must accept 'signal' in the method signature.
    def forward(self, x: torch.Tensor, t: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        # The 'signal' argument from the solver is the conditioning vector.
        # We pass it to our actual model, which expects it as 'condition'.
        return self.model(x, t, condition=signal)
    # --- END FIX ---


def sample_parameters(model, condition, device, x_0=None, num_sample=1000):
    """Generate parameters for a given *conditioning vector* using ODE solver sampling"""
    condition = condition.float()
    if condition.shape[0] != 1:
        raise ValueError("Only one conditioning vector (from one signal) can be passed at a time")
    
    # Repeat the condition for parallel sampling
    repeated_condition = condition.repeat(num_sample, 1) # [num_sample, conditioning_dim]
    
    # Initialize x_0 if not provided
    if x_0 is None:
        x_0 = torch.randn(num_sample, 3, device=device) # param_dim is 3
    else:
        x_0 = x_0.to(device).float()

    wrapped_model = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Sample using the solver
    # This call passes 'signal=repeated_condition' to the ODESolver,
    # which in turn passes it to our WrappedModel.forward method.
    samples = solver.sample(
        time_grid=torch.tensor([0.0, 1.0], device=device).float(),
        x_init=x_0,
        method='dopri5',
        step_size=None,
        signal=repeated_condition  # Pass the repeated condition as 'signal'
    )
    
    return samples

def analyze_posterior(model, condition, true_params, parameters_std, parameters_mean, 
                     n_samples=1000, device="cuda"):
    """Generate and analyze parameter posterior distribution from the condition"""
    # Process in batches to avoid GPU memory issues
    batch_size = 500  # Can be larger as the model is smaller
    all_params = []
    num_batches = n_samples // batch_size + (1 if n_samples % batch_size else 0)
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        batch_params = sample_parameters(
            model, condition, device, 
            num_sample=current_batch_size
        )
        all_params.append(batch_params.cpu())
    
    # Combine batches
    generated_params = torch.cat(all_params, dim=0).numpy()
    
    # Denormalize parameters
    generated_params = generated_params * parameters_std + parameters_mean
    
    # Convert true_params to numpy without scaling
    true_params = true_params.cpu().numpy().squeeze()
    
    # Create visualization
    labels = ['lnAmplitude', 'lnFrequency', 'lnFrequency_dot']
    fig = corner.corner(
        generated_params,
        labels=labels,
        truths=true_params,
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
    
    plt.suptitle("Parameter Posterior Distribution (Two-Stage Model)\n" + true_values_text, 
                 y=1.02, fontsize=10)
    
    return generated_params, fig