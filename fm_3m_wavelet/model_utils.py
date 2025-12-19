import torch.nn as nn

def print_model_structure(model, filename=None):
    """
    Prints and optionally saves a detailed view of the model's structure
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
    output.append("\nContinuousFlowMatcher Configuration:")
    output.append(f"├─ Parameter Dimension: {model.param_dim}")
    output.append(f"├─ Hidden Dimension: {model.hidden_dim}")
    output.append(f"├─ Signal Input Dimension: {model.signal_input_dim}")
    
    # Flow model structure
    output.append("\nFlow Model Structure:")
    output.extend(format_module(model.flow_model))
    
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
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Model Structure:\n")
            f.write(formatted_output)
    
    return formatted_output