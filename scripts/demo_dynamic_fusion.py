#!/usr/bin/env python
# scripts/demo_dynamic_fusion.py
"""
Demo script for visualizing the effect of dynamic fusion on Vietnamese OCR.
Shows how dynamic fusion helps capture both local and global features for diacritical
mark recognition.
"""

import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import LinearSegmentedColormap
import logging

from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from utils.ctc_utils import CTCDecoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('demo_dynamic_fusion')

def visualize_fusion_weights(model, image_path, output_dir=None):
    """
    Visualize the dynamic fusion weights for an input image.
    Shows how different encoder layers contribute to the final features.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.processor(image, return_tensors="pt").pixel_values.to(device)
    
    # We'll need to get internal weights from dynamic fusion module
    # Store original forward method
    original_dynamic_fusion_forward = model.dynamic_fusion.forward
    
    # Extracted weights
    global_weights = None
    
    # Create a hook to capture the fusion weights
    def hook_dynamic_fusion(features_list):
        # Get global weights
        batch_size = features_list[0].shape[0]
        global_features = []
        for features in features_list:
            # Global pooling to get a representation of the entire sequence
            mean_features = torch.mean(features, dim=1)
            global_features.append(mean_features)
        
        # Concatenate global features from all layers
        concatenated_global = torch.cat(global_features, dim=-1)
        
        # Generate global fusion weights
        nonlocal global_weights
        global_weights = model.dynamic_fusion.context_encoder(concatenated_global)
        global_weights = torch.nn.functional.softmax(global_weights, dim=-1)
        
        # Continue with the original forward pass
        return original_dynamic_fusion_forward(features_list)
    
    # Replace forward method with our hooked version
    model.dynamic_fusion.forward = hook_dynamic_fusion
    
    # Perform a forward pass (inference only)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs['logits']
        
        # Decode the prediction
        if hasattr(model, 'combined_char_vocab'):
            idx_to_char = {i: c for i, c in enumerate(model.combined_char_vocab)}
            blank_idx = 0  # Assuming blank is at index 0
            decoder = CTCDecoder(idx_to_char_map=idx_to_char, blank_idx=blank_idx)
            predicted_text = decoder(logits)[0]
            logger.info(f"Predicted text: {predicted_text}")
    
    # Restore original forward method
    model.dynamic_fusion.forward = original_dynamic_fusion_forward
    
    # Visualize global weights
    if global_weights is not None:
        weights_np = global_weights.cpu().numpy()[0]  # First batch item
        
        # Create a bar chart of the global weights
        plt.figure(figsize=(10, 6))
        indices = range(len(weights_np))
        bars = plt.bar(indices, weights_np)
        
        # Annotate with values
        for i, v in enumerate(weights_np):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.title('Dynamic Fusion: Layer Importance Weights')
        plt.xlabel('Encoder Layer Index')
        plt.ylabel('Weight')
        plt.ylim(0, max(weights_np) + 0.1)
        plt.xticks(indices)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "dynamic_fusion_weights.png"))
            logger.info(f"Saved weights visualization to {output_dir}/dynamic_fusion_weights.png")
        else:
            plt.show()
        
        plt.close()
    
    # Return the predicted text for reference
    return predicted_text

def visualize_with_and_without_fusion(model, image_path, output_dir=None):
    """
    Compare model predictions with and without dynamic fusion.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Check if dynamic fusion is enabled
    if not (hasattr(model, 'dynamic_fusion') and model.dynamic_fusion is not None):
        logger.warning("Model does not have dynamic fusion enabled!")
        return None
    
    # Create a decoder
    if hasattr(model, 'combined_char_vocab'):
        idx_to_char = {i: c for i, c in enumerate(model.combined_char_vocab)}
        blank_idx = 0  # Assuming blank is at index 0
        decoder = CTCDecoder(idx_to_char_map=idx_to_char, blank_idx=blank_idx)
    else:
        logger.error("Model does not have character vocabulary information!")
        return None
    
    # First get prediction with dynamic fusion (default model behavior)
    model.eval()
    with torch.no_grad():
        outputs_with_fusion = model(pixel_values=pixel_values)
        logits_with_fusion = outputs_with_fusion['logits']
        prediction_with_fusion = decoder(logits_with_fusion)[0]
    
    # Now disable dynamic fusion by replacing it temporarily
    dynamic_fusion_module = model.dynamic_fusion
    
    # Replace with a basic fusion method (e.g., just take the last layer or average)
    def simple_fusion(features_list):
        # Simple averaging of features
        stacked_features = torch.stack(features_list, dim=0)
        return torch.mean(stacked_features, dim=0)
    
    # Create a temporary module with forward method
    class SimpleFusion(torch.nn.Module):
        def forward(self, features_list):
            return simple_fusion(features_list)
    
    # Replace the dynamic fusion module
    model.dynamic_fusion = SimpleFusion()
    
    # Get prediction without dynamic fusion
    with torch.no_grad():
        outputs_without_fusion = model(pixel_values=pixel_values)
        logits_without_fusion = outputs_without_fusion['logits']
        prediction_without_fusion = decoder(logits_without_fusion)[0]
    
    # Restore the original dynamic fusion module
    model.dynamic_fusion = dynamic_fusion_module
    
    # Create a comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Display original image
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Create a text comparison panel
    plt.subplot(2, 1, 2)
    plt.axis('off')
    
    comparison_text = (
        f"With Dynamic Fusion: {prediction_with_fusion}\n\n"
        f"Without Dynamic Fusion: {prediction_without_fusion}"
    )
    
    # Calculate character differences 
    diff_chars = []
    min_len = min(len(prediction_with_fusion), len(prediction_without_fusion))
    
    for i in range(min_len):
        if prediction_with_fusion[i] != prediction_without_fusion[i]:
            diff_chars.append((i, prediction_with_fusion[i], prediction_without_fusion[i]))
    
    # Add difference highlighting
    if diff_chars:
        comparison_text += "\n\nDifferences:\n"
        for pos, char_with, char_without in diff_chars:
            context_start = max(0, pos - 5)
            context_end = min(len(prediction_with_fusion), pos + 6)
            context = prediction_with_fusion[context_start:context_end]
            highlight_pos = pos - context_start
            
            highlighted_context = (
                f"Position {pos}: '{context[:highlight_pos]}"
                f"[{char_with}]"
                f"{context[highlight_pos+1:]}' vs '{char_without}'"
            )
            comparison_text += f"{highlighted_context}\n"
    
    plt.text(0.1, 0.8, comparison_text, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "fusion_comparison.png"))
        logger.info(f"Saved comparison visualization to {output_dir}/fusion_comparison.png")
    else:
        plt.show()
    
    plt.close()
    
    return {
        "with_fusion": prediction_with_fusion,
        "without_fusion": prediction_without_fusion,
        "differences": diff_chars
    }

def main():
    parser = argparse.ArgumentParser(description="Demo for Dynamic Multi-Scale Fusion in OCR")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained hierarchical model with dynamic fusion")
    parser.add_argument("--combined_char_vocab_path", type=str, required=True,
                        help="Path to combined character vocabulary JSON file")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to Vietnamese handwritten text image to analyze")
    parser.add_argument("--output_dir", type=str, default="dynamic_fusion_demo_output",
                        help="Directory to save visualization outputs")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    try:
        logger.info(f"Loading vocabulary from {args.combined_char_vocab_path}")
        with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f:
            combined_vocab = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        return 1
    
    # Load model
    try:
        logger.info(f"Loading model from {args.model_path}")
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
            args.model_path,
            combined_char_vocab=combined_vocab
        )
        model.to(device)
        
        # Check if the model has dynamic fusion enabled
        has_dynamic_fusion = hasattr(model, 'dynamic_fusion') and model.dynamic_fusion is not None
        if not has_dynamic_fusion:
            logger.warning("This model does not have dynamic fusion enabled! Demo may not work correctly.")
        
        # Log which fusion features are enabled
        logger.info("Model configuration:")
        if hasattr(model.config, 'use_dynamic_fusion'):
            logger.info(f"Dynamic Multi-Scale Fusion: {'Enabled' if model.config.use_dynamic_fusion else 'Disabled'}")
        if hasattr(model.config, 'use_feature_enhancer'):
            logger.info(f"Local Feature Enhancer: {'Enabled' if model.config.use_feature_enhancer else 'Disabled'}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run visualizations
    logger.info(f"Processing image: {args.image_path}")
    
    try:
        # Visualize fusion weights
        visualize_fusion_weights(model, args.image_path, args.output_dir)
        
        # Compare with and without fusion
        comparison_results = visualize_with_and_without_fusion(model, args.image_path, args.output_dir)
        
        if comparison_results and comparison_results['differences']:
            logger.info("Found differences in predictions with and without dynamic fusion:")
            for pos, char_with, char_without in comparison_results['differences']:
                logger.info(f"  Position {pos}: '{char_with}' vs '{char_without}'")
        
        logger.info(f"Demo completed. Visualizations saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)