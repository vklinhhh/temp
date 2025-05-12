# scripts/visualize_single_image.py

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import json
import logging

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder
except ImportError:
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizeSingle')

# def save_attention_visualization(
#     original_pil_image,
#     attention_map_1d,
#     vision_encoder_config,
#     processor_target_size,
#     output_path_prefix # Will save multiple files based on this prefix
#     # REMOVED: predicted_text argument
# ):
#     saved_files = []
#     try:
#         if not hasattr(vision_encoder_config, 'patch_size'):
#              logger.error("Vision encoder config missing 'patch_size'. Cannot visualize attention.")
#              return None
             
#         if isinstance(attention_map_1d, torch.Tensor):
#             attention_map_1d = attention_map_1d.cpu().numpy()

#         patch_size = vision_encoder_config.patch_size
#         processed_height, processed_width = processor_target_size
#         num_patches_h = processed_height // patch_size
#         num_patches_w = processed_width // patch_size

#         expected_num_patches = num_patches_h * num_patches_w
#         if attention_map_1d.shape[0] == expected_num_patches + 1:
#             logger.debug("Attention map includes CLS token. Slicing it off.")
#             attention_map_patches = attention_map_1d[1:, :]
#         elif attention_map_1d.shape[0] == expected_num_patches:
#             attention_map_patches = attention_map_1d
#         else:
#             logger.warning(f"Mismatch in attention map length ({attention_map_1d.shape[0]}) vs expected ({expected_num_patches} or {expected_num_patches+1}). Skipping viz.")
#             return None

#         output_dir = os.path.dirname(output_path_prefix)
#         if output_dir:
#              os.makedirs(output_dir, exist_ok=True)

#         original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
#         img_h, img_w = original_cv_image.shape[:2]
#         region_names = ["Above", "Middle", "Below"]

#         for i in range(attention_map_patches.shape[1]):
#             region_attention_1d = attention_map_patches[:, i]
#             try:
#                 attention_grid_2d = region_attention_1d.reshape(num_patches_h, num_patches_w)
#             except ValueError as reshape_err:
#                 logger.error(f"Error reshaping attention for region {region_names[i]}: {reshape_err}. Skipping region.")
#                 continue

#             heatmap_resized_raw = cv2.resize(attention_grid_2d, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
#             heatmap_normalized = cv2.normalize(heatmap_resized_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#             heatmap_normalized_uint8 = heatmap_normalized.astype(np.uint8)
#             heatmap_color = cv2.applyColorMap(heatmap_normalized_uint8, cv2.COLORMAP_JET)
#             overlayed_image = cv2.addWeighted(original_cv_image, 0.6, heatmap_color, 0.4, 0)

#             # --- TEXT OVERLAYS REMOVED ---
#             # font = cv2.FONT_HERSHEY_SIMPLEX
#             # text_size, _ = cv2.getTextSize(f"Pred: {predicted_text}", font, 0.7, 2)
#             # cv2.rectangle(overlayed_image, (5, 5), (10 + text_size[0], 40), (0, 0, 0), -1)
#             # cv2.putText(overlayed_image, f"Pred: {predicted_text}", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
#             #
#             # text_size_region, _ = cv2.getTextSize(f"Region: {region_names[i]}", font, 0.7, 2)
#             # cv2.rectangle(overlayed_image, (5, img_h - 30), (10 + text_size_region[0], img_h - 5), (0, 0, 0), -1)
#             # cv2.putText(overlayed_image, f"Region: {region_names[i]}", (10, img_h - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
#             # --- END TEXT OVERLAYS REMOVED ---

#             base_name = os.path.basename(output_path_prefix)
#             output_filename = os.path.join(output_dir, f"{base_name}_region{i}_{region_names[i]}.png")
#             cv2.imwrite(output_filename, overlayed_image)
#             saved_files.append(output_filename)
#             logger.debug(f"Saved attention visualization to {output_filename}")
#         return saved_files
#     except Exception as e:
#         logger.error(f"Error generating attention visualization: {e}", exc_info=True)
#         return None

# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # --- Load Vocabulary ---
#     try:
#         logger.info(f"Loading combined char vocab from: {args.combined_char_vocab_path}")
#         with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f:
#             combined_char_vocab = json.load(f)
#         if not combined_char_vocab: raise ValueError("Combined vocab empty.")
#         combined_char_to_idx = {c: i for i, c in enumerate(combined_char_vocab)}
#         combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
#         blank_idx = combined_char_to_idx.get('<blank>', 0)
#         logger.info(f"Combined vocab loaded: {len(combined_char_vocab)} chars.")
#     except Exception as e:
#         logger.error(f"FATAL: Failed to load vocabulary: {e}", exc_info=True)
#         return

#     # --- Load Model and Processor ---
#     try:
#         logger.info(f"Loading trained Hierarchical CTC model from: {args.model_path}")
#         model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
#             args.model_path,
#             combined_char_vocab=combined_char_vocab
#         )
#         processor = model.processor
#         model.to(device)
#         model.eval()
#         logger.info("Model and processor loaded.")

#         has_visual_attention = getattr(model.config, 'use_visual_diacritic_attention', False)
#         if has_visual_attention:
#              logger.info("Visual Diacritic Attention is enabled in this model.")
#         else:
#              logger.info("Visual Diacritic Attention is NOT enabled. No attention maps will be generated.")
             
#         processor_target_size = None
#         if hasattr(processor, 'size') and isinstance(processor.size, dict):
#              processor_target_size = (processor.size.get("height"), processor.size.get("width"))
#         elif hasattr(model.config, 'vision_encoder_config') and hasattr(model.config.vision_encoder_config, 'image_size'):
#              img_s = model.config.vision_encoder_config.image_size
#              if isinstance(img_s, int): processor_target_size = (img_s, img_s)
        
#         if processor_target_size is None or None in processor_target_size:
#              logger.warning("Could not determine processor target size. Using default (384, 384). This might be incorrect.")
#              processor_target_size = (384, 384)
#         else:
#              logger.info(f"Processor target size: {processor_target_size}")

#     except Exception as e:
#         logger.error(f"FATAL: Failed to load model: {e}", exc_info=True)
#         return

#     # --- Load and Preprocess Image ---
#     try:
#         logger.info(f"Loading image from: {args.image_path}")
#         original_image_pil = Image.open(args.image_path).convert("RGB")
#         pixel_values = processor(original_image_pil, return_tensors="pt").pixel_values
#         pixel_values = pixel_values.to(device)
#     except FileNotFoundError:
#         logger.error(f"FATAL: Image file not found at {args.image_path}")
#         return
#     except UnidentifiedImageError:
#          logger.error(f"FATAL: Cannot identify or open image file at {args.image_path}")
#          return
#     except Exception as e:
#         logger.error(f"FATAL: Error processing image: {e}", exc_info=True)
#         return

#     # --- Run Inference ---
#     logger.info("Running inference...")
#     attention_maps = None
#     predicted_text = "<ERROR>"
#     try:
#         with torch.no_grad():
#             outputs = model(
#                 pixel_values=pixel_values,
#                 return_diacritic_attention=has_visual_attention
#             )
#             logits = outputs.get('logits')

#             if logits is not None:
#                 ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
#                 decoded_preds = ctc_decoder(logits)
#                 predicted_text = decoded_preds[0] if decoded_preds else "<EMPTY>"
#                 logger.info(f"Predicted text: '{predicted_text}'") # Still log it to console

#                 if has_visual_attention:
#                     attention_maps = outputs.get('visual_diacritic_attention_maps')
#                     if attention_maps is not None:
#                          logger.info("Received visual diacritic attention maps.")
#                     else:
#                          logger.warning("Requested attention maps, but they were not found in model output.")
#             else:
#                 logger.error("Model output did not contain 'logits'.")
#     except Exception as e:
#         logger.error(f"Error during model inference: {e}", exc_info=True)

#     # --- Visualize and Save ---
#     if has_visual_attention and attention_maps is not None:
#         logger.info(f"Generating attention visualization(s) with prefix: {args.output_path}")
#         attention_map_single = attention_maps[0]
        
#         output_prefix = args.output_path
#         if output_prefix.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             output_prefix = os.path.splitext(output_prefix)[0]

#         saved_files = save_attention_visualization( # REMOVED predicted_text from call
#             original_pil_image=original_image_pil,
#             attention_map_1d=attention_map_single,
#             vision_encoder_config=model.config.vision_encoder_config,
#             processor_target_size=processor_target_size,
#             output_path_prefix=output_prefix
#         )

#         if saved_files:
#             logger.info(f"Successfully saved attention visualization(s): {', '.join(saved_files)}")
#         else:
#             logger.error("Failed to generate or save attention visualization.")
#             # Fallback: Save just the original image (no text)
#             try:
#                  output_dir = os.path.dirname(args.output_path)
#                  base_name = os.path.basename(args.output_path)
#                  fallback_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_original.png")
#                  original_image_pil.save(fallback_path)
#                  logger.info(f"Saved original image (no text, no attention) to: {fallback_path}")
#             except Exception as save_err:
#                  logger.error(f"Failed to save fallback original image: {save_err}")
#     else:
#         logger.info("Attention visualization not generated (not enabled or inference/viz failed).")
#         # Fallback: Save just the original image (no text)
#         try:
#             output_dir = os.path.dirname(args.output_path)
#             base_name = os.path.basename(args.output_path)
#             if output_dir: os.makedirs(output_dir, exist_ok=True)
#             output_filename = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_original.png")
#             original_image_pil.save(output_filename)
#             logger.info(f"Saved original image (no text, no attention) to: {output_filename}")
#         except Exception as save_err:
#             logger.error(f"Failed to save original image: {save_err}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize prediction and diacritic attention for a single image.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to trained Hierarchical CTC model directory")
#     parser.add_argument("--combined_char_vocab_path", type=str, required=True, help="Path to COMBINED char vocab JSON")
#     parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
#     parser.add_argument("--output_path", type=str, required=True, help="Path prefix for the output visualization PNG file(s). E.g., 'output/viz_image'.")
    
#     args = parser.parse_args()
#     main(args)


# scripts/visualize_single_image.py

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F # For ReLU in Grad-CAM
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import json
import logging

# --- Imports for model and utils ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder
except ImportError:
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizeSingle')


# --- Grad-CAM Generation Function ---
def generate_grad_cam(model, original_pil_image, pixel_values_tensor,
                      processor_target_size, vision_encoder_config,
                      target_diacritic_index,
                      target_token_index=None,
                      target_layer_name="shared_layer"
                     ):
    model.eval()
    model.zero_grad()
    model.clear_hooks() # Clear any previous hooks

    if not pixel_values_tensor.requires_grad:
         pixel_values_tensor.requires_grad_(True) # Ensure input allows grad computation

    outputs = model(pixel_values=pixel_values_tensor, grad_cam_target_layer_module_name=target_layer_name)
    grad_cam_logits = outputs.get('grad_cam_target_logits')

    if grad_cam_logits is None:
        logger.error("Could not find 'grad_cam_target_logits' in model output.")
        model.clear_hooks()
        return None

    if target_token_index is not None and target_token_index < grad_cam_logits.shape[1]:
        score = grad_cam_logits[0, target_token_index, target_diacritic_index]
    else:
        score = grad_cam_logits[0, :, target_diacritic_index].mean()

    try:
        score.backward()
    except RuntimeError as e:
        logger.error(f"RuntimeError during backward pass for Grad-CAM: {e}. Check if the graph was retained or if a tensor that does not require grad is involved.", exc_info=True)
        model.clear_hooks()
        return None


    activations, gradients = model.get_activations_and_gradients()

    if activations is None or gradients is None:
        logger.error(f"Failed to get activations or gradients for Grad-CAM from layer '{target_layer_name}'.")
        model.clear_hooks()
        return None

    # activations and gradients are already detached in get_activations_and_gradients
    
    pooled_gradients = torch.mean(gradients, dim=[1]) # Assuming gradients shape [1, seq_len, feature_dim] -> [1, feature_dim]
    
    activations_squeezed = activations.squeeze(0) # [seq_len, feature_dim]
    pooled_gradients_squeezed = pooled_gradients.squeeze(0) # [feature_dim]
    
    # Ensure float type for einsum
    activations_squeezed = activations_squeezed.float()
    pooled_gradients_squeezed = pooled_gradients_squeezed.float()

    heatmap_1d = torch.einsum('sf,f->s', activations_squeezed, pooled_gradients_squeezed)
    heatmap_1d = F.relu(heatmap_1d)

    try:
        if not hasattr(vision_encoder_config, 'patch_size'): raise ValueError("patch_size missing from vision_encoder_config")
        patch_size = vision_encoder_config.patch_size
        processed_height, processed_width = processor_target_size
        num_patches_h = processed_height // patch_size
        num_patches_w = processed_width // patch_size
        expected_num_patches = num_patches_h * num_patches_w
        
        activation_seq_len = activations.shape[1] # This is the sequence length of the hooked layer

        if activation_seq_len == expected_num_patches + 1: # e.g., output of ViT encoder
            heatmap_patches = heatmap_1d[1:]
            logger.debug(f"Grad-CAM: Hooked layer output length {activation_seq_len}, assuming CLS token, using elements 1: for spatial map.")
        elif activation_seq_len == expected_num_patches: # e.g., output of fusion layer if it drops CLS
            heatmap_patches = heatmap_1d
            logger.debug(f"Grad-CAM: Hooked layer output length {activation_seq_len} matches expected patches.")
        else:
            logger.warning(f"Grad-CAM: Activation length ({activation_seq_len}) from hooked layer '{target_layer_name}' does not directly match expected spatial patches ({expected_num_patches}). This might lead to incorrect reshaping if the layer doesn't output spatially corresponding features (e.g. if it's a global pool). Assuming it's spatially correspondent for now if it can be reshaped.")
            if activation_seq_len == num_patches_h * num_patches_w: # Check if it's already flat spatial
                 heatmap_patches = heatmap_1d
            else:
                logger.error(f"Cannot reliably reshape heatmap from activation length {activation_seq_len} to {num_patches_h}x{num_patches_w}.")
                model.clear_hooks(); return None
        
        cam_2d = heatmap_patches.reshape(num_patches_h, num_patches_w).cpu().numpy()
    except Exception as reshape_e:
        logger.error(f"Error reshaping Grad-CAM (activation shape: {activations.shape if activations is not None else 'None'}, heatmap_1d shape: {heatmap_1d.shape}): {reshape_e}", exc_info=True)
        model.clear_hooks(); return None

    cam_2d = cv2.resize(cam_2d, (original_pil_image.width, original_pil_image.height), interpolation=cv2.INTER_LINEAR)
    cam_2d = np.maximum(cam_2d, 0)
    max_val = np.max(cam_2d)
    if max_val > 1e-9: cam_2d = cam_2d / max_val
    else: cam_2d = np.zeros_like(cam_2d)

    model.clear_hooks()
    if pixel_values_tensor.grad is not None: pixel_values_tensor.grad.zero_()
    return cam_2d

# --- Function to Save Grad-CAM Overlay ---
def save_grad_cam_visualization(original_pil_image, grad_cam_map, output_path, alpha=0.6):
    if grad_cam_map is None: logger.warning("Grad-CAM map is None, cannot save."); return
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
        logger.info(f"Saved Grad-CAM visualization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Grad-CAM visualization to {output_path}: {e}", exc_info=True)

# --- Main Execution Logic ---
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab = json.load(f)
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
        blank_idx = combined_char_vocab.index('<blank>') if '<blank>' in combined_char_vocab else 0
    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}", exc_info=True); return

    try:
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(args.model_path, combined_char_vocab=combined_char_vocab)
        processor = model.processor
        model.to(device); model.eval()
        processor_target_size = (getattr(processor, 'size', {}).get("height", 384), getattr(processor, 'size', {}).get("width", 384))
        if hasattr(model.config, 'vision_encoder_config') and model.config.vision_encoder_config and hasattr(model.config.vision_encoder_config, 'image_size'):
             img_s = model.config.vision_encoder_config.image_size
             if isinstance(img_s, int): processor_target_size = (img_s, img_s)
        logger.info(f"Processor target size for CAM: {processor_target_size}")
    except Exception as e: logger.error(f"FATAL: Model load fail: {e}", exc_info=True); return

    try:
        original_image_pil = Image.open(args.image_path).convert("RGB")
        pixel_values = processor(original_image_pil, return_tensors="pt").pixel_values.to(device)
    except Exception as e: logger.error(f"FATAL: Image process fail: {e}", exc_info=True); return

    predicted_text = "<ERROR_PRED>"
    try:
        with torch.no_grad():
            outputs_pred = model(pixel_values=pixel_values.clone()) # Use a clone for non-grad prediction
            logits_pred = outputs_pred.get('logits')
            if logits_pred is not None:
                ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
                predicted_text = ctc_decoder(logits_pred)[0]
        logger.info(f"Predicted text: '{predicted_text}'")
    except Exception as pred_e: logger.error(f"Error during prediction inference: {pred_e}", exc_info=True)

    if args.grad_cam_target_diacritic:
        target_diacritic_name = args.grad_cam_target_diacritic
        diacritic_vocab_list = getattr(model.config, 'diacritic_vocab', [])
        if not diacritic_vocab_list: logger.error("Diacritic vocab not in model.config for Grad-CAM.")
        else:
            try:
                target_diacritic_index = diacritic_vocab_list.index(target_diacritic_name)
                logger.info(f"Generating Grad-CAM for diacritic '{target_diacritic_name}' (idx {target_diacritic_index}) on layer '{args.grad_cam_target_layer}'...")
                
                grad_cam_map = generate_grad_cam(
                    model, original_image_pil, pixel_values.clone(), # Pass a clone for grad_cam processing
                    processor_target_size, model.config.vision_encoder_config,
                    target_diacritic_index, args.grad_cam_token_index, args.grad_cam_target_layer
                )
                if grad_cam_map is not None:
                    base_out = os.path.splitext(args.output_path)[0]
                    token_s = f"_token{args.grad_cam_token_index}" if args.grad_cam_token_index is not None else "_agg"
                    grad_cam_out_path = f"{base_out}_gradcam_{target_diacritic_name}_{args.grad_cam_target_layer.replace('.', '-')}{token_s}.png"
                    save_grad_cam_visualization(original_image_pil, grad_cam_map, grad_cam_out_path, args.grad_cam_alpha)
                else: logger.error("Grad-CAM map generation failed.")
            except ValueError: logger.error(f"Diacritic '{target_diacritic_name}' not found in model's diacritic_vocab.")
            except Exception as gc_e: logger.error(f"Error during Grad-CAM: {gc_e}", exc_info=True)
    else:
        logger.info("No target diacritic for Grad-CAM. Saving prediction overlay only.")
        try:
            out_dir = os.path.dirname(args.output_path); base_name = os.path.basename(args.output_path)
            if out_dir: os.makedirs(out_dir, exist_ok=True)
            pred_only_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_pred_only.png")
            img_cv = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(img_cv, f"Pred: {predicted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA) # Black outline
            cv2.putText(img_cv, f"Pred: {predicted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA) # White text
            cv2.imwrite(pred_only_path, img_cv)
            logger.info(f"Saved image with prediction text to: {pred_only_path}")
        except Exception as save_e: logger.error(f"Failed to save prediction-only image: {save_e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize prediction and Grad-CAM for a single image.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--combined_char_vocab_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Path prefix for output files.")
    parser.add_argument("--grad_cam_target_diacritic", type=str, default=None, help="Target diacritic name (e.g., 'acute').")
    parser.add_argument("--grad_cam_token_index", type=int, default=None, help="Optional token index for Grad-CAM.")
    parser.add_argument("--grad_cam_target_layer", type=str, default="shared_layer", help="Model layer for Grad-CAM (e.g., 'shared_layer', 'dynamic_fusion', 'transformer_encoder.layers.5').")
    parser.add_argument("--grad_cam_alpha", type=float, default=0.6, help="Blending alpha for Grad-CAM overlay.")
    args = parser.parse_args()
    main(args)