# utils/compatibility_logging.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import logging

logger = logging.getLogger(__name__)

def log_compatibility_matrix(model, epoch, step, output_dir, log_interval=1000):
    """
    Log the CharacterDiacriticCompatibility matrix during training.
    
    Args:
        model: The model containing the compatibility module
        epoch: Current epoch
        step: Current optimization step
        output_dir: Directory to save visualizations
        log_interval: How often to log (in steps)
    """
    if step % log_interval != 0:
        return
    
    # Check if model has compatibility module
    if not hasattr(model, 'character_diacritic_compatibility') or model.character_diacritic_compatibility is None:
        return
    
    compat_module = model.character_diacritic_compatibility
    compat_matrix = compat_module.compatibility_matrix.detach().cpu()
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "compatibility_matrix_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get vocabularies for axis labels
    base_vocab = compat_module.base_char_vocab if hasattr(compat_module, 'base_char_vocab') else None
    diac_vocab = compat_module.diacritic_vocab if hasattr(compat_module, 'diacritic_vocab') else None
    
    # Save raw matrix data for later analysis
    # np.save(os.path.join(viz_dir, f"compat_matrix_e{epoch}_s{step}.npy"), compat_matrix.numpy())
    
    # Generate and save visualization
    visualize_compatibility_matrix(
        compat_matrix, 
        base_vocab, 
        diac_vocab, 
        os.path.join(viz_dir, f"compat_matrix_e{epoch}_s{step}.png"),
        title=f"Compatibility Matrix - Epoch {epoch}, Step {step}"
    )
    
    # Log statistics
    if 'wandb' in globals() and wandb.run:
        try:
            # Create a simplified heatmap for wandb
            fig, _ = plt.subplots(figsize=(10, 8))
            visualize_compatibility_matrix(compat_matrix, base_vocab, diac_vocab, return_fig=True, fig=fig)
            wandb.log({
                "compatibility_matrix/epoch": epoch,
                "compatibility_matrix/step": step,
                "compatibility_matrix/mean": compat_matrix.mean().item(),
                "compatibility_matrix/std": compat_matrix.std().item(),
                "compatibility_matrix/min": compat_matrix.min().item(),
                "compatibility_matrix/max": compat_matrix.max().item(),
                "compatibility_matrix/visualization": wandb.Image(fig)
            })
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Error logging compatibility matrix to wandb: {e}")


def visualize_compatibility_matrix(
    compat_matrix, 
    base_vocab=None, 
    diac_vocab=None, 
    output_path=None,
    title="Character-Diacritic Compatibility Matrix",
    return_fig=False,
    fig=None
):
    """
    Create a visualization of the compatibility matrix.
    
    Args:
        compat_matrix: The compatibility matrix tensor
        base_vocab: List of base characters (for axis labels)
        diac_vocab: List of diacritics (for axis labels)
        output_path: Where to save the visualization (if None, just display)
        title: Title for the plot
        return_fig: Whether to return the figure object
        fig: Existing figure to use (if None, create new)
    
    Returns:
        fig, ax if return_fig=True, otherwise None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    
    # Convert to numpy array if it's a tensor
    if isinstance(compat_matrix, torch.Tensor):
        compat_matrix = compat_matrix.cpu().numpy()
    
    # Create custom colormap (better for distinguishing positive/negative compatibility)
    colors = [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.8, 0.0)]
    cmap = LinearSegmentedColormap.from_list("compatibility_cmap", colors, N=256)
    
    # Get matrix dimensions
    n_chars, n_diacritics = compat_matrix.shape
    
    # Create figure if not provided
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        ax = fig.gca()
    
    # Create heatmap
    sns.heatmap(
        compat_matrix, 
        cmap=cmap,
        center=0,
        annot=False, # Don't show numbers in cells (too many)
        fmt=".1f", 
        linewidths=0.5,
        ax=ax,
        xticklabels=diac_vocab if diac_vocab else [str(i) for i in range(n_diacritics)],
        yticklabels=base_vocab if base_vocab else [str(i) for i in range(n_chars)],
        vmin=-3.0,  # Clip very negative values
        vmax=3.0,   # Clip very positive values
    )
    
    # Add labels
    plt.title(title, fontsize=14)
    plt.xlabel("Diacritics", fontsize=12)
    plt.ylabel("Base Characters", fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved compatibility matrix visualization to {output_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
        
        if not return_fig:
            plt.close(fig)
    
    if return_fig:
        return fig, ax
    return None


# utils/compatibility_logging.py

def track_compatibility_matrix_gradients(model, epoch, step, output_dir, log_interval=1000):
    """Track the gradients flowing to compatibility matrix during training."""
    if step % log_interval != 0:
        return
    
    if not hasattr(model, 'character_diacritic_compatibility') or model.character_diacritic_compatibility is None:
        return
    
    compat_module = model.character_diacritic_compatibility
    compat_matrix = compat_module.compatibility_matrix
    
    # Check if matrix has gradients
    if compat_matrix.grad is None:
        print(f"WARNING: Compatibility matrix has no gradients at step {step}!")
        return
    
    # Get gradient statistics
    grad_mean = compat_matrix.grad.abs().mean().item()
    grad_max = compat_matrix.grad.abs().max().item()
    
    print(f"Compatibility Matrix Gradients - Step {step} - Mean: {grad_mean:.6f}, Max: {grad_max:.6f}")
    
    # Save gradient data
    grad_dir = os.path.join(output_dir, "compatibility_gradients")
    os.makedirs(grad_dir, exist_ok=True)
    
    # Save gradient statistics to a CSV file
    import csv
    csv_path = os.path.join(grad_dir, "gradient_stats.csv")
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'step', 'grad_mean', 'grad_max'])
        writer.writerow([epoch, step, grad_mean, grad_max])
    
    # Save raw gradient data (optional, can be large)
    # np.save(os.path.join(grad_dir, f"compat_matrix_grad_e{epoch}_s{step}.npy"), 
    #         compat_matrix.grad.detach().cpu().numpy())


def log_compatibility_effects_during_training(model, base_logits, raw_diacritic_logits, 
                                            enhanced_diacritic_logits, epoch, step, 
                                            output_dir, log_interval=1000):
    """Log the actual effects of compatibility matrix during training"""
    if step % log_interval != 0:
        return
    
    if not hasattr(model, 'character_diacritic_compatibility') or model.character_diacritic_compatibility is None:
        return
    
    # Calculate compatibility bias
    compatibility_bias = enhanced_diacritic_logits - raw_diacritic_logits
    
    # Get base character predictions
    base_preds = torch.argmax(base_logits, dim=-1)
    
    # Analyze bias effects for specific characters
    effects_log = []
    
    for sample_idx in range(min(2, base_logits.shape[0])):
        for time_idx in range(min(5, base_logits.shape[1])):
            base_char_idx = base_preds[sample_idx, time_idx].item()
            base_char = model.base_char_vocab[base_char_idx]
            
            bias_vector = compatibility_bias[sample_idx, time_idx]
            
            # Find most boosted and suppressed diacritics
            top_k = 3
            boosted_indices = torch.topk(bias_vector, k=top_k).indices
            suppressed_indices = torch.topk(bias_vector, k=top_k, largest=False).indices
            
            boosted_diacritics = [(model.diacritic_vocab[idx.item()], bias_vector[idx].item()) 
                                for idx in boosted_indices]
            suppressed_diacritics = [(model.diacritic_vocab[idx.item()], bias_vector[idx].item()) 
                                   for idx in suppressed_indices]
            
            effects_log.append({
                'base_char': base_char,
                'boosted': boosted_diacritics,
                'suppressed': suppressed_diacritics
            })
    
    # Save to file
    effects_file = os.path.join(output_dir, "compatibility_effects", f"effects_e{epoch}_s{step}.json")
    os.makedirs(os.path.dirname(effects_file), exist_ok=True)
    
    with open(effects_file, 'w', encoding='utf-8') as f:
        json.dump(effects_log, f, ensure_ascii=False, indent=2)
    
    # Log summary to console
    print(f"Compatibility Effects Summary (Epoch {epoch}, Step {step}):")
    for effect in effects_log[:3]:  # Show first 3 examples
        print(f"  Base '{effect['base_char']}' → Boosted: {effect['boosted'][:2]}")
        print(f"  Base '{effect['base_char']}' → Suppressed: {effect['suppressed'][:2]}")