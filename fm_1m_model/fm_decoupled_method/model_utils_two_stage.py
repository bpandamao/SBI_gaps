"""
Utility functions for printing model structures in the two-stage decoupled method.
"""

import torch.nn as nn


def print_dcae_structure(model, filename=None):
    """
    Prints and optionally saves a detailed view of the DCAE model's structure.
    
    Args:
        model: The DCAE model instance
        filename: Optional filename to save the structure to
    """
    output = []
    
    def format_module(module, prefix=""):
        module_str = []
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                module_str.append(f"{prefix}├─ {name}:")
                for i, sub_module in enumerate(child):
                    if isinstance(sub_module, nn.Dropout):
                        module_str.append(f"{prefix}│  ├─ {i}: Dropout(p={sub_module.p})")
                    elif isinstance(sub_module, (nn.BatchNorm1d, nn.LayerNorm)):
                        module_str.append(f"{prefix}│  ├─ {i}: {sub_module.__class__.__name__}")
                    elif isinstance(sub_module, nn.Linear):
                        module_str.append(f"{prefix}│  ├─ {i}: Linear({sub_module.in_features}, {sub_module.out_features})")
                    elif isinstance(sub_module, (nn.Conv1d, nn.ConvTranspose1d)):
                        module_str.append(f"{prefix}│  ├─ {i}: {sub_module.__class__.__name__}(in={sub_module.in_channels}, out={sub_module.out_channels}, kernel={sub_module.kernel_size[0]})")
                    else:
                        module_str.append(f"{prefix}│  ├─ {i}: {sub_module.__class__.__name__}")
            elif isinstance(child, nn.ModuleList):
                module_str.append(f"{prefix}├─ {name}: ModuleList")
                for i, sub_module in enumerate(child):
                    module_str.extend(format_module(sub_module, prefix + "│  "))
            elif isinstance(child, nn.Linear):
                module_str.append(f"{prefix}├─ {name}: Linear({child.in_features}, {child.out_features})")
            elif isinstance(child, (nn.Conv1d, nn.ConvTranspose1d)):
                module_str.append(f"{prefix}├─ {name}: {child.__class__.__name__}(in={child.in_channels}, out={child.out_channels}, kernel={child.kernel_size[0]})")
            else:
                module_str.append(f"{prefix}├─ {name}: {child.__class__.__name__}")
        return module_str

    # Model configuration
    output.append("\nDCAE Model Configuration:")
    output.append(f"├─ Bottleneck Dimension: {model.bottleneck_dim}")
    output.append(f"├─ Signal Length: {model.decoder.signal_length}")
    
    # Encoder structure
    output.append("\nEncoder Structure:")
    output.extend(format_module(model.encoder))
    
    # Decoder structure
    output.append("\nDecoder Structure:")
    output.extend(format_module(model.decoder))
    
    # Model Parameters
    output.append("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    output.append(f"Total parameters: {total_params:,}\n")
    
    for name, param in model.named_parameters():
        output.append(f"{name}: {list(param.shape)}")
    
    # Join all lines
    formatted_output = "\n".join(output)
    
    # Print to console
    print(formatted_output)
    
    # Save to file if filename is provided
    if filename:
        with open(filename, 'w') as f:
            f.write("DCAE Model Structure:\n")
            f.write(formatted_output)
    
    return formatted_output


def print_flow_bottleneck_structure(model, filename=None):
    """
    Prints and optionally saves a detailed view of the Flow Matcher Bottleneck model's structure.
    
    Args:
        model: The ContinuousFlowMatcherBottleneck model instance
        filename: Optional filename to save the structure to
    """
    output = []
    
    def format_module(module, prefix=""):
        module_str = []
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                module_str.append(f"{prefix}├─ {name}:")
                for i, sub_module in enumerate(child):
                    if isinstance(sub_module, nn.Dropout):
                        module_str.append(f"{prefix}│  ├─ {i}: Dropout(p={sub_module.p})")
                    elif isinstance(sub_module, (nn.BatchNorm1d, nn.LayerNorm)):
                        module_str.append(f"{prefix}│  ├─ {i}: {sub_module.__class__.__name__}")
                    elif isinstance(sub_module, nn.Linear):
                        module_str.append(f"{prefix}│  ├─ {i}: Linear({sub_module.in_features}, {sub_module.out_features})")
                    else:
                        module_str.append(f"{prefix}│  ├─ {i}: {sub_module.__class__.__name__}")
            elif isinstance(child, nn.ModuleList):
                module_str.append(f"{prefix}├─ {name}: ModuleList")
                for i, sub_module in enumerate(child):
                    module_str.extend(format_module(sub_module, prefix + "│  "))
            elif isinstance(child, nn.Linear):
                module_str.append(f"{prefix}├─ {name}: Linear({child.in_features}, {child.out_features})")
            else:
                module_str.append(f"{prefix}├─ {name}: {child.__class__.__name__}")
        return module_str

    # Model configuration
    output.append("\nFlow Matcher Bottleneck Configuration:")
    output.append(f"├─ Parameter Dimension: {model.param_dim}")
    output.append(f"├─ Conditioning Dimension: {model.conditioning_dim}")
    output.append(f"├─ Hidden Dimension: {model.hidden_dim}")
    output.append(f"├─ Time Embedding Dimension: {model.time_embed_dim}")
    
    # Model structure
    output.append("\nModel Structure:")
    output.extend(format_module(model))
    
    # Model Parameters
    output.append("\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    output.append(f"Total parameters: {total_params:,}\n")
    
    for name, param in model.named_parameters():
        output.append(f"{name}: {list(param.shape)}")
    
    # Join all lines
    formatted_output = "\n".join(output)
    
    # Print to console
    print(formatted_output)
    
    # Save to file if filename is provided
    if filename:
        with open(filename, 'w') as f:
            f.write("Flow Matcher Bottleneck Model Structure:\n")
            f.write(formatted_output)
    
    return formatted_output

