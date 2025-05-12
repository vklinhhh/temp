# scripts/visualize_all_layers_grad_cam.py

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
import json
import logging
import matplotlib.pyplot as plt # For combining images
import torch.nn as nn
# --- Imports for model and utils ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder
    # Import the Grad-CAM functions from visualize_single_image.py
    # To avoid code duplication, ensure visualize_single_image.py is importable
    # or copy the necessary functions here. For simplicity, let's assume it's importable.
    # If not, you'd copy generate_grad_cam and save_grad_cam_visualization here.
    sys.path.append(os.path.dirname(__file__)) # Add current script's dir to path
    from visualize_single_image import generate_grad_cam, save_grad_cam_visualization
except ImportError as e:
    print(f"ImportError: {e}. Make sure visualize_single_image.py is in the same directory or PYTHONPATH is set.")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizeAllLayers')

def combine_grad_cam_images(image_paths, output_combined_path, titles, main_title="Grad-CAM Across Layers", font_path=None):
    """
    Combines multiple Grad-CAM images into a single grid image using Matplotlib.
    """
    if not image_paths:
        logger.warning("No images to combine.")
        return

    num_images = len(image_paths)
    # Determine grid size (e.g., try to make it somewhat square)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5 + 1)) # +1 for main title
    axes = axes.flatten() # Flatten in case of single row/col

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(titles[i] if i < len(titles) else os.path.basename(img_path))
            axes[i].axis('off')
        except FileNotFoundError:
            logger.error(f"Image not found: {img_path}")
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[i].axis('off')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            axes[i].text(0.5, 0.5, 'Error loading', ha='center', va='center')
            axes[i].axis('off')


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if font_path:
        try:
            prop = ImageFont.truetype(font_path, 16)
            fig.suptitle(main_title, fontsize=20, fontproperties=prop if prop else None)
        except IOError:
            logger.warning(f"Font not found at {font_path}, using default.")
            fig.suptitle(main_title, fontsize=20)
    else:
        fig.suptitle(main_title, fontsize=20)


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_combined_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_combined_path)
    plt.close(fig) # Close the figure to free memory
    logger.info(f"Combined Grad-CAM visualization saved to {output_combined_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load Vocabulary ---
    try:
        with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab = json.load(f)
        # combined_char_to_idx = {c: i for i, c in enumerate(combined_char_vocab)} # Not directly used here
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
        blank_idx = combined_char_vocab.index('<blank>') if '<blank>' in combined_char_vocab else 0
    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}"); return

    # --- Load Model and Processor ---
    try:
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(args.model_path, combined_char_vocab=combined_char_vocab)
        processor = model.processor
        model.to(device); model.eval()
        processor_target_size = (getattr(processor, 'size', {}).get("height", 384), getattr(processor, 'size', {}).get("width", 384))
        if hasattr(model.config, 'vision_encoder_config') and model.config.vision_encoder_config and hasattr(model.config.vision_encoder_config, 'image_size'):
             img_s = model.config.vision_encoder_config.image_size
             if isinstance(img_s, int): processor_target_size = (img_s, img_s)
        logger.info(f"Processor target size for CAM: {processor_target_size}")
    except Exception as e: logger.error(f"FATAL: Model load fail: {e}"); return

    # --- Load Image ---
    try:
        original_image_pil = Image.open(args.image_path).convert("RGB")
        pixel_values = processor(original_image_pil, return_tensors="pt").pixel_values.to(device)
    except Exception as e: logger.error(f"FATAL: Image process fail: {e}"); return

    # --- Get Prediction (once) ---
    predicted_text = "<ERROR_PRED>"
    try:
        with torch.no_grad():
            outputs_pred = model(pixel_values=pixel_values.clone())
            logits_pred = outputs_pred.get('logits')
            if logits_pred is not None:
                ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
                predicted_text = ctc_decoder(logits_pred)[0]
        logger.info(f"Predicted text for {os.path.basename(args.image_path)}: '{predicted_text}'")
    except Exception as pred_e: logger.error(f"Error during prediction: {pred_e}")

    # --- Define Candidate Layers for Grad-CAM ---
    # You should customize this list based on your model structure (from named_modules())
    # and what you want to investigate.
    candidate_layers = [
        "vision_encoder.encoder.layer.5",  # Middle ViT layer
        "vision_encoder.encoder.layer.11", # Last ViT layer (assuming 12 layers)
        # Add your fusion layer name if it exists and is not None
    ]
    if model.fusion_projection is not None: candidate_layers.append("fusion_projection")
    if model.dynamic_fusion is not None: candidate_layers.append("dynamic_fusion")
    if model.feature_enhancer is not None: candidate_layers.append("feature_enhancer")
    candidate_layers.append("pos_encoder") # If it's a module, not nn.Identity
    candidate_layers.extend([f"transformer_encoder.layers.{i}" for i in [0, model.config.num_transformer_encoder_layers -1 ]]) # First and Last custom TF layer
    candidate_layers.append("shared_layer")

    # Filter out layers that might be None if certain features are disabled
    valid_candidate_layers = []
    for layer_name in candidate_layers:
        try:
            current_obj = model
            for part in layer_name.split('.'):
                if part.isdigit(): current_obj = current_obj[int(part)]
                else: current_obj = getattr(current_obj, part)
            if current_obj is not None and isinstance(current_obj, nn.Module):
                valid_candidate_layers.append(layer_name)
            else:
                logger.warning(f"Skipping candidate layer '{layer_name}' as it's None or not an nn.Module.")
        except (AttributeError, IndexError, ValueError):
            logger.warning(f"Skipping candidate layer '{layer_name}' as it could not be resolved.")
    
    logger.info(f"Will attempt Grad-CAM for layers: {valid_candidate_layers}")


    generated_cam_paths = []
    cam_titles = []

    # --- Loop Through Layers and Generate Grad-CAM ---
    if args.grad_cam_target_diacritic:
        target_diacritic_name = args.grad_cam_target_diacritic
        diacritic_vocab_list = getattr(model.config, 'diacritic_vocab', [])
        if not diacritic_vocab_list:
            logger.error("Diacritic vocabulary not in model.config. Cannot perform Grad-CAM.")
            return
        try:
            target_diacritic_index = diacritic_vocab_list.index(target_diacritic_name)
        except ValueError:
            logger.error(f"Diacritic '{target_diacritic_name}' not found in model's diacritic_vocab: {diacritic_vocab_list}")
            return

        base_output_filename = os.path.splitext(os.path.basename(args.output_path_prefix))[0]
        output_dir_for_layers = os.path.join(os.path.dirname(args.output_path_prefix), f"{base_output_filename}_all_layers_gradcam")
        os.makedirs(output_dir_for_layers, exist_ok=True)

        for i, layer_name in enumerate(valid_candidate_layers):
            logger.info(f"--- Generating Grad-CAM for layer ({i+1}/{len(valid_candidate_layers)}): '{layer_name}' ---")
            try:
                grad_cam_map = generate_grad_cam(
                    model, original_image_pil, pixel_values.clone(),
                    processor_target_size, model.config.vision_encoder_config,
                    target_diacritic_index, args.grad_cam_token_index, layer_name
                )
                if grad_cam_map is not None:
                    layer_name_safe = layer_name.replace('.', '_') # For filename
                    token_s = f"_token{args.grad_cam_token_index}" if args.grad_cam_token_index is not None else "_agg"
                    individual_cam_path = os.path.join(output_dir_for_layers,
                                                       f"gradcam_{target_diacritic_name}_{layer_name_safe}{token_s}.png")
                    save_grad_cam_visualization(
                        original_image_pil, grad_cam_map, individual_cam_path, args.grad_cam_alpha
                    )
                    generated_cam_paths.append(individual_cam_path)
                    cam_titles.append(f"Layer: {layer_name}")
                else:
                    logger.warning(f"Grad-CAM map generation failed for layer '{layer_name}'.")
            except Exception as e_layer_cam:
                logger.error(f"Error generating Grad-CAM for layer '{layer_name}': {e_layer_cam}", exc_info=True)
        
        # --- Combine Images ---
        if generated_cam_paths:
            combined_output_path = os.path.join(os.path.dirname(args.output_path_prefix), f"{base_output_filename}_gradcam_all_layers_for_{target_diacritic_name}.png")
            main_plot_title = f"Grad-CAM for Diacritic '{target_diacritic_name}' (Image: {os.path.basename(args.image_path)})"
            combine_grad_cam_images(generated_cam_paths, combined_output_path, cam_titles, main_plot_title)
        else:
            logger.info("No Grad-CAM images were generated to combine.")
    else:
        logger.info("No target diacritic specified. Skipping Grad-CAM generation for all layers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for multiple layers for a single image.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--combined_char_vocab_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path_prefix", type=str, required=True, help="Path prefix for output files. A subdirectory will be created based on this.")
    parser.add_argument("--grad_cam_target_diacritic", type=str, required=True, help="Target diacritic name (e.g., 'acute').")
    parser.add_argument("--grad_cam_token_index", type=int, default=None, help="Optional token index for Grad-CAM.")
    parser.add_argument("--grad_cam_alpha", type=float, default=0.6, help="Blending alpha for Grad-CAM overlay.")
    # No --grad_cam_target_layer here, as we loop through a predefined list.
    
    args = parser.parse_args()
    main(args)