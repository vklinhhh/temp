# scripts/visualize_single_image.py
# ... (imports and other functions like _reshape_and_resize_cam, save_grad_cam_visualization, generate_grad_cam, generate_grad_cam_plus_plus remain the same) ...
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import json
import logging
import math # For _reshape_and_resize_cam

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder
except ImportError:
    print("Warning: Could not import model/utils relative to project root. Attempting direct import.")
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizeSingle')

def _reshape_and_resize_cam(cam_1d, original_pil_image, vision_encoder_config, processor_target_size, hooked_layer_name, activation_seq_len):
    # Ensure vision_encoder_config and patch_size are valid
    if not vision_encoder_config or not hasattr(vision_encoder_config, 'patch_size') or not vision_encoder_config.patch_size:
        logger.warning(f"CAM Reshape: vision_encoder_config or patch_size missing/invalid for layer '{hooked_layer_name}'. Attempting to infer patch grid if possible, or defaulting.")
        # Fallback: if it's a square-ish number of activations, try to make a grid.
        # This is highly heuristic and might not be correct for all layers.
        side = int(math.sqrt(activation_seq_len))
        if side * side == activation_seq_len:
            num_patches_h, num_patches_w = side, side
            logger.debug(f"CAM Reshape: Inferred square patch grid {num_patches_h}x{num_patches_w} for layer '{hooked_layer_name}'.")
            cam_patches = cam_1d
        else: # Cannot infer a simple grid, this will likely fail or be distorted if layer isn't already 2D-like
            raise ValueError(f"Cannot reliably reshape CAM for layer '{hooked_layer_name}' with activation length {activation_seq_len} without valid patch_size.")
    else:
        patch_size = vision_encoder_config.patch_size
        processed_height, processed_width = processor_target_size
        num_patches_h = processed_height // patch_size
        num_patches_w = processed_width // patch_size
        expected_num_patches = num_patches_h * num_patches_w
        cam_patches = None

        if activation_seq_len == expected_num_patches + 1:
            cam_patches = cam_1d[1:]
            logger.debug(f"CAM Reshape: Hooked layer '{hooked_layer_name}' output length {activation_seq_len}, assuming CLS token.")
        elif activation_seq_len == expected_num_patches:
            cam_patches = cam_1d
            logger.debug(f"CAM Reshape: Hooked layer '{hooked_layer_name}' output length {activation_seq_len} matches expected patches.")
        elif activation_seq_len == num_patches_h * num_patches_w : # Already flat spatial
            cam_patches = cam_1d
        else:
            raise ValueError(f"CAM Reshape: Activation length ({activation_seq_len}) from hooked layer '{hooked_layer_name}' "
                             f"does not match expected spatial patches ({expected_num_patches} or {expected_num_patches+1}).")

    if cam_patches is None or cam_patches.numel() != num_patches_h * num_patches_w:
        raise ValueError(f"CAM Reshape: Resulting cam_patches length ({cam_patches.numel() if cam_patches is not None else 'None'}) "
                         f"does not match expected reshaped patches ({num_patches_h * num_patches_w}) for layer '{hooked_layer_name}'. "
                         f"Original activation_seq_len: {activation_seq_len}")

    cam_2d = cam_patches.reshape(num_patches_h, num_patches_w).cpu().numpy()
    cam_resized = cv2.resize(cam_2d, (original_pil_image.width, original_pil_image.height), interpolation=cv2.INTER_LINEAR)
    return cam_resized


# generate_grad_cam, generate_grad_cam_plus_plus remain the same as previously provided
# ... (assuming they are correctly defined as in the previous full response) ...
def generate_grad_cam(model, original_pil_image, pixel_values_tensor,
                      processor_target_size, vision_encoder_config,
                      target_class_index,
                      target_token_index=None,
                      target_layer_name="shared_layer",
                      grad_cam_logit_target_type="diacritic" 
                     ):
    model.eval(); model.zero_grad(); model.clear_hooks() 
    if not pixel_values_tensor.requires_grad: pixel_values_tensor.requires_grad_(True)
    outputs = model(pixel_values=pixel_values_tensor,
                    grad_cam_target_layer_module_name=target_layer_name,
                    grad_cam_logit_target_type=grad_cam_logit_target_type)
    grad_cam_logits = outputs.get('grad_cam_target_logits')
    if grad_cam_logits is None or grad_cam_logits.numel() == 0:
        logger.error(f"[GradCAM] Empty 'grad_cam_target_logits' (type: {grad_cam_logit_target_type})."); model.clear_hooks(); return None
    if target_class_index >= grad_cam_logits.shape[-1]:
        logger.error(f"[GradCAM] target_class_index {target_class_index} OOB for logits shape {grad_cam_logits.shape}"); model.clear_hooks(); return None
    
    if target_token_index is not None and target_token_index < grad_cam_logits.shape[1]: score = grad_cam_logits[0, target_token_index, target_class_index]
    elif target_token_index is not None: logger.warning(f"[GradCAM] token_index {target_token_index} OOB. Using mean."); score = grad_cam_logits[0, :, target_class_index].mean()
    else: score = grad_cam_logits[0, :, target_class_index].mean()

    try: score.backward()
    except RuntimeError as e: logger.error(f"[GradCAM] backward error: {e}.", exc_info=True); model.clear_hooks(); return None
    activations, gradients = model.get_activations_and_gradients()
    if activations is None or gradients is None: logger.error(f"[GradCAM] No activations/gradients from '{target_layer_name}'."); model.clear_hooks(); return None
    
    pooled_gradients = torch.mean(gradients, dim=[1]) 
    act_sq = activations.squeeze(0).float(); pool_grad_sq = pooled_gradients.squeeze(0).float()
    if act_sq.ndim == 2 and pool_grad_sq.ndim == 1 and act_sq.shape[1] == pool_grad_sq.shape[0]: heatmap_1d = torch.einsum('sf,f->s', act_sq, pool_grad_sq)
    else: logger.error(f"[GradCAM] Mismatched shapes: act {act_sq.shape}, grad_pool {pool_grad_sq.shape}"); model.clear_hooks(); return None
    heatmap_1d = F.relu(heatmap_1d)
    try: cam_2d_resized = _reshape_and_resize_cam(heatmap_1d, original_pil_image, vision_encoder_config, processor_target_size, target_layer_name, activations.shape[1])
    except ValueError as e: logger.error(f"[GradCAM] Reshape error for '{target_layer_name}': {e}"); model.clear_hooks(); return None
    cam_2d_resized = np.maximum(cam_2d_resized, 0); max_val = np.max(cam_2d_resized)
    if max_val > 1e-9: cam_2d_resized = cam_2d_resized / max_val
    else: cam_2d_resized = np.zeros_like(cam_2d_resized)
    model.clear_hooks()
    if pixel_values_tensor.grad is not None: pixel_values_tensor.grad.zero_()
    return cam_2d_resized

def generate_grad_cam_plus_plus(model, original_pil_image, pixel_values_tensor,
                                processor_target_size, vision_encoder_config,
                                target_class_index,
                                target_token_index=None,
                                target_layer_name="shared_layer",
                                grad_cam_logit_target_type="diacritic"
                               ):
    model.eval(); model.zero_grad(); model.clear_hooks()
    if not pixel_values_tensor.requires_grad: pixel_values_tensor.requires_grad_(True)
    outputs = model(pixel_values=pixel_values_tensor, grad_cam_target_layer_module_name=target_layer_name, grad_cam_logit_target_type=grad_cam_logit_target_type)
    grad_cam_logits = outputs.get('grad_cam_target_logits')
    if grad_cam_logits is None or grad_cam_logits.numel() == 0: logger.error(f"[GradCAM++] Empty 'grad_cam_target_logits' (type: {grad_cam_logit_target_type})."); model.clear_hooks(); return None
    if target_class_index >= grad_cam_logits.shape[-1]: logger.error(f"[GradCAM++] target_class_index {target_class_index} OOB."); model.clear_hooks(); return None
    
    if target_token_index is not None and target_token_index < grad_cam_logits.shape[1]: score = grad_cam_logits[0, target_token_index, target_class_index]
    elif target_token_index is not None: logger.warning(f"[GradCAM++] token_index {target_token_index} OOB. Using mean."); score = grad_cam_logits[0, :, target_class_index].mean()
    else: score = grad_cam_logits[0, :, target_class_index].mean()
    try: score.backward(retain_graph=True)
    except RuntimeError as e: logger.error(f"[GradCAM++] backward error: {e}.", exc_info=True); model.clear_hooks(); return None
    activations, gradients = model.get_activations_and_gradients()
    if activations is None or gradients is None: logger.error(f"[GradCAM++] No activations/gradients from '{target_layer_name}'."); model.clear_hooks(); return None

    grads_power_2 = gradients.pow(2); grads_power_3 = gradients.pow(3)
    sum_act_grad_pow_3 = (activations * grads_power_3).sum(dim=2, keepdim=True)
    alpha_denom = 2 * grads_power_2 + sum_act_grad_pow_3
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * 1e-8)
    alpha = grads_power_2 / alpha_denom
    weights = (alpha * F.relu(gradients)).sum(dim=1, keepdim=True)
    cam_1d = (activations * weights).sum(dim=2).squeeze(0)
    cam_1d = F.relu(cam_1d)
    try: cam_2d_resized = _reshape_and_resize_cam(cam_1d, original_pil_image, vision_encoder_config, processor_target_size, target_layer_name, activations.shape[1])
    except ValueError as e: logger.error(f"[GradCAM++] Reshape error for '{target_layer_name}': {e}"); model.clear_hooks(); return None
    cam_2d_resized = np.maximum(cam_2d_resized, 0); max_val = np.max(cam_2d_resized)
    if max_val > 1e-9: cam_2d_resized = cam_2d_resized / max_val
    else: cam_2d_resized = np.zeros_like(cam_2d_resized)
    model.clear_hooks()
    if pixel_values_tensor.grad is not None: pixel_values_tensor.grad.zero_()
    return cam_2d_resized


### ADDED FUNCTION: generate_layer_cam ###
def generate_layer_cam(model, original_pil_image, pixel_values_tensor,
                       processor_target_size, vision_encoder_config,
                       target_class_index,
                       target_token_index=None,
                       target_layer_name="shared_layer",
                       grad_cam_logit_target_type="diacritic"
                      ):
    model.eval()
    model.zero_grad()
    model.clear_hooks()

    if not pixel_values_tensor.requires_grad:
        pixel_values_tensor.requires_grad_(True)

    outputs = model(pixel_values=pixel_values_tensor,
                    grad_cam_target_layer_module_name=target_layer_name,
                    grad_cam_logit_target_type=grad_cam_logit_target_type)
    grad_cam_logits = outputs.get('grad_cam_target_logits')

    if grad_cam_logits is None or grad_cam_logits.numel() == 0:
        logger.error(f"[LayerCAM] Could not find or received empty 'grad_cam_target_logits' (type: {grad_cam_logit_target_type}).")
        model.clear_hooks(); return None
    
    if target_class_index >= grad_cam_logits.shape[-1]:
        logger.error(f"[LayerCAM] target_class_index {target_class_index} is out of bounds for logits shape {grad_cam_logits.shape}")
        model.clear_hooks(); return None

    if target_token_index is not None and target_token_index < grad_cam_logits.shape[1]:
        score = grad_cam_logits[0, target_token_index, target_class_index]
    elif target_token_index is not None:
        logger.warning(f"[LayerCAM] target_token_index {target_token_index} is out of bounds. Using mean over sequence.")
        score = grad_cam_logits[0, :, target_class_index].mean()
    else:
        score = grad_cam_logits[0, :, target_class_index].mean()

    try:
        score.backward()
    except RuntimeError as e:
        logger.error(f"[LayerCAM] RuntimeError during backward pass: {e}.", exc_info=True)
        model.clear_hooks(); return None

    activations, gradients = model.get_activations_and_gradients()

    if activations is None or gradients is None:
        logger.error(f"[LayerCAM] Failed to get activations or gradients from layer '{target_layer_name}'.")
        model.clear_hooks(); return None

    # Layer-CAM specific calculation: element-wise product of activations and positive gradients
    # activations: [B, S, F], gradients: [B, S, F]
    # We want sum_F (A_sf * ReLU(grad_sf)) for each s
    
    # In LayerCAM paper, weights are directly the positive gradients: w_k = ReLU(gradient_k)
    # Then L_LayerCAM = ReLU ( sum_k w_k * A_k )
    # Here, A_k is feature map k. So, A_sf is activation at seq_pos s, feature f.
    # grad_sf is gradient dS/dA_sf.
    
    # So, layer_cam_weights = F.relu(gradients) # [B,S,F]
    # cam_1d = (activations * layer_cam_weights).sum(dim=2) # Sum over feature dimension F -> [B,S]
    # cam_1d = F.relu(cam_1d.squeeze(0)) # Squeeze batch, apply final ReLU -> [S]
    
    # Alternative formulation often seen: just sum the positive part of (act * grad) over features
    cam_1d = (activations * F.relu(gradients)).sum(dim=2).squeeze(0) # Sum over F. Result [S]
    cam_1d = F.relu(cam_1d) # Final ReLU on the sum


    try:
        cam_2d_resized = _reshape_and_resize_cam(cam_1d, original_pil_image, vision_encoder_config, processor_target_size, target_layer_name, activations.shape[1])
    except ValueError as reshape_e:
        logger.error(f"[LayerCAM] Error reshaping for layer '{target_layer_name}': {reshape_e}", exc_info=True)
        model.clear_hooks(); return None

    cam_2d_resized = np.maximum(cam_2d_resized, 0)
    max_val = np.max(cam_2d_resized)
    if max_val > 1e-9: cam_2d_resized = cam_2d_resized / max_val
    else: cam_2d_resized = np.zeros_like(cam_2d_resized)

    model.clear_hooks()
    if pixel_values_tensor.grad is not None: pixel_values_tensor.grad.zero_()
    return cam_2d_resized


def save_grad_cam_visualization(original_pil_image, grad_cam_map, output_path, alpha=0.6):
    # ... (same as before)
    if grad_cam_map is None: logger.warning("CAM map is None, cannot save."); return
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        original_cv = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        original_cv = np.float32(original_cv) / 255.0
        if heatmap.shape[0:2] != original_cv.shape[0:2]:
            heatmap = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
        overlay = heatmap * alpha + original_cv * (1 - alpha)
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, overlay)
        logger.info(f"Saved CAM visualization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CAM visualization to {output_path}: {e}", exc_info=True)


# Standalone main for visualize_single_image.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize prediction and CAM for a single image.")
    # ... (all arguments from previous version of this file's main)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--combined_char_vocab_path", type=str, required=True)
    parser.add_argument("--base_char_vocab_path", type=str) # Optional, but needed for 'base' logit target
    parser.add_argument("--diacritic_vocab_path", type=str) # Optional, but needed for 'diacritic' logit target
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path prefix for output files.")
    
    parser.add_argument("--cam_method", type=str, default="gradcam", choices=["gradcam", "gradcampp", "layercam"], help="CAM method to use.")
    parser.add_argument("--grad_cam_logit_target", type=str, default="diacritic", choices=["final", "diacritic", "base"], help="Which logits to target for CAM.")
    
    parser.add_argument("--grad_cam_target_item", type=str, required=True, help="Target item name (char for 'final'/'base', diacritic name for 'diacritic').")
    
    parser.add_argument("--grad_cam_token_index", type=int, default=None, help="Optional token index for CAM.")
    parser.add_argument("--grad_cam_target_layer", type=str, default="shared_layer", help="Model layer for CAM.")
    parser.add_argument("--grad_cam_alpha", type=float, default=0.6, help="Blending alpha for CAM overlay.")
    args = parser.parse_args()
    
    # Simplified main logic for testing this script directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(args.model_path, combined_char_vocab_path=args.combined_char_vocab_path) # Adapt as needed
    processor = model.processor
    model.to(device).eval()
    
    original_image_pil = Image.open(args.image_path).convert("RGB")
    pixel_values = processor(original_image_pil, return_tensors="pt").pixel_values.to(device)

    # Determine processor_target_size (simplified for this example)
    processor_target_size = (processor.size.get("height",384), processor.size.get("width",384)) if hasattr(processor, "size") else (384,384)
    if hasattr(model.config, 'vision_encoder_config') and model.config.vision_encoder_config and hasattr(model.config.vision_encoder_config, 'image_size'):
        img_s = model.config.vision_encoder_config.image_size
        processor_target_size = (img_s,img_s) if isinstance(img_s, int) else tuple(img_s) if isinstance(img_s, (list,tuple)) else processor_target_size


    # Determine target_class_index based on logit_target and target_item
    target_class_index = -1
    if args.grad_cam_logit_target == "final":
        with open(args.combined_char_vocab_path, 'r') as f: vocab = json.load(f)
        target_class_index = vocab.index(args.grad_cam_target_item)
    elif args.grad_cam_logit_target == "base":
        with open(args.base_char_vocab_path, 'r') as f: vocab = json.load(f)
        target_class_index = vocab.index(args.grad_cam_target_item)
    elif args.grad_cam_logit_target == "diacritic":
        with open(args.diacritic_vocab_path, 'r') as f: vocab = json.load(f)
        target_class_index = vocab.index(args.grad_cam_target_item)

    cam_map = None
    if args.cam_method == "gradcampp":
        cam_map = generate_grad_cam_plus_plus(model, original_image_pil, pixel_values, processor_target_size, model.config.vision_encoder_config, target_class_index, args.grad_cam_token_index, args.grad_cam_target_layer, args.grad_cam_logit_target)
    elif args.cam_method == "layercam":
        cam_map = generate_layer_cam(model, original_image_pil, pixel_values, processor_target_size, model.config.vision_encoder_config, target_class_index, args.grad_cam_token_index, args.grad_cam_target_layer, args.grad_cam_logit_target)
    else:
        cam_map = generate_grad_cam(model, original_image_pil, pixel_values, processor_target_size, model.config.vision_encoder_config, target_class_index, args.grad_cam_token_index, args.grad_cam_target_layer, args.grad_cam_logit_target)

    if cam_map is not None:
        save_grad_cam_visualization(original_image_pil, cam_map, args.output_path, args.grad_cam_alpha)
    else:
        logger.error("CAM generation failed in standalone test.")