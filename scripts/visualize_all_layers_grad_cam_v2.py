# scripts/visualize_all_layers_grad_cam.py
# ... (imports, combine_grad_cam_images function remain the same)
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
import matplotlib.pyplot as plt
import torch.nn as nn
import math

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
    from utils.ctc_utils import CTCDecoder
    sys.path.append(os.path.dirname(__file__))
    from visualize_single_image import (
        generate_grad_cam, generate_grad_cam_plus_plus, generate_layer_cam, # ADDED generate_layer_cam
        save_grad_cam_visualization
    )
except ImportError as e:
    print(f"ImportError: {e}.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizeAllLayers')

def combine_grad_cam_images(image_paths, output_combined_path, titles, main_title="Grad-CAM Across Layers", font_path=None):
    # ... (same as previous full version)
    if not image_paths: logger.warning("No images to combine."); return
    num_images = len(image_paths); cols = int(np.ceil(np.sqrt(num_images))); rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5 + 1.5)); axes = axes.flatten()
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(titles[i] if i < len(titles) else os.path.basename(img_path), fontsize=10)
            axes[i].axis('off')
        except Exception as e: logger.error(f"Err loading img {img_path}: {e}"); axes[i].text(0.5,0.5,'Err'); axes[i].axis('off')
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    import matplotlib.font_manager
    try: mpl_font_props = matplotlib.font_manager.FontProperties(fname=font_path, size=16) if font_path else None
    except Exception: mpl_font_props = None; logger.warning(f"Could not use font {font_path}, using default.")
    fig.suptitle(main_title, fontproperties=mpl_font_props, fontsize=16 if not mpl_font_props else None)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]); output_dir = os.path.dirname(output_combined_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_combined_path); plt.close(fig)
    logger.info(f"Combined CAM visualization saved to {output_combined_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load Vocabularies ---
    try:
        with open(args.combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab_list = json.load(f)
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab_list)}
        combined_char_to_idx = {c: i for i,c in enumerate(combined_char_vocab_list)}
        blank_idx = combined_char_vocab_list.index('<blank>') if '<blank>' in combined_char_vocab_list else 0
        
        base_char_vocab_list = None
        base_char_to_idx = None
        if args.base_char_vocab_path:
            with open(args.base_char_vocab_path, 'r', encoding='utf-8') as f: base_char_vocab_list = json.load(f)
            if base_char_vocab_list: base_char_to_idx = {c: i for i,c in enumerate(base_char_vocab_list)}
        
        diacritic_vocab_list = None
        diacritic_to_idx = None
        if args.diacritic_vocab_path:
             with open(args.diacritic_vocab_path, 'r', encoding='utf-8') as f: diacritic_vocab_list = json.load(f)
             if diacritic_vocab_list: diacritic_to_idx = {c:i for i,c in enumerate(diacritic_vocab_list)}

    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}", exc_info=True); return

    # --- Load Model and Processor ---
    try:
        model_kwargs = {'combined_char_vocab': combined_char_vocab_list}
        if base_char_vocab_list: model_kwargs['base_char_vocab'] = base_char_vocab_list
        if diacritic_vocab_list: model_kwargs['diacritic_vocab'] = diacritic_vocab_list
        
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(args.model_path, **model_kwargs)
        processor = model.processor
        model.to(device); model.eval()

        if hasattr(processor, 'size') and isinstance(processor.size, dict) and \
           processor.size.get("height") is not None and processor.size.get("width") is not None:
            processor_target_size = (processor.size["height"], processor.size["width"])
        elif hasattr(model.config, 'vision_encoder_config') and \
             model.config.vision_encoder_config and \
             hasattr(model.config.vision_encoder_config, 'image_size'):
            img_s = model.config.vision_encoder_config.image_size
            processor_target_size = (img_s, img_s) if isinstance(img_s, int) else tuple(img_s) if isinstance(img_s, (list, tuple)) and len(img_s) == 2 else (384,384)
        else: 
            processor_target_size = (384, 384) 
        logger.info(f"Processor target size for CAM: {processor_target_size}")

    except Exception as e: logger.error(f"FATAL: Model load fail: {e}", exc_info=True); return

    # --- Load Image ---
    try:
        original_image_pil = Image.open(args.image_path).convert("RGB")
        pixel_values = processor(original_image_pil, return_tensors="pt").pixel_values.to(device)
    except Exception as e: logger.error(f"FATAL: Image process fail: {e}", exc_info=True); return

    # --- Get Prediction ---
    predicted_text = "<ERROR_PRED>"
    raw_logits_for_char_search = None # Logits corresponding to args.grad_cam_logit_target
    with torch.no_grad():
        outputs_pred = model(pixel_values=pixel_values.clone(), grad_cam_logit_target_type=args.grad_cam_logit_target)
        
        final_ctc_logits = outputs_pred.get('logits') 
        if final_ctc_logits is not None and final_ctc_logits.numel() > 0 :
            ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
            predicted_text = ctc_decoder(final_ctc_logits)[0]
        logger.info(f"Predicted text for {os.path.basename(args.image_path)}: '{predicted_text}'")

        if args.grad_cam_logit_target == "final": raw_logits_for_char_search = outputs_pred.get('logits')
        elif args.grad_cam_logit_target == "base": raw_logits_for_char_search = outputs_pred.get('base_logits')
        elif args.grad_cam_logit_target == "diacritic": raw_logits_for_char_search = outputs_pred.get('diacritic_logits')
        else: raw_logits_for_char_search = final_ctc_logits

    # --- Determine Target Index for CAM ---
    actual_grad_cam_token_index = args.grad_cam_token_index
    target_class_index = -1
    class_name_for_log = "N/A"
    focused_item_info = ""
    target_item_name = args.grad_cam_target_item # Use the unified target item arg

    current_vocab_map_for_target = None
    if args.grad_cam_logit_target == "final":
        current_vocab_map_for_target = combined_char_to_idx
        class_name_for_log = f"char '{target_item_name}' in combined_vocab"
    elif args.grad_cam_logit_target == "base":
        current_vocab_map_for_target = base_char_to_idx
        class_name_for_log = f"base_char '{target_item_name}' in base_vocab"
    elif args.grad_cam_logit_target == "diacritic":
        current_vocab_map_for_target = diacritic_to_idx
        class_name_for_log = f"diacritic '{target_item_name}' in diacritic_vocab"

    if not target_item_name:
        logger.error(f"FATAL: No target item specified (--grad_cam_target_item) for CAM with logit target '{args.grad_cam_logit_target}'.")
        return
    if not current_vocab_map_for_target:
        logger.error(f"FATAL: Vocabulary for logit target '{args.grad_cam_logit_target}' is not available/loaded. Check vocab paths.")
        return

    try:
        target_class_index = current_vocab_map_for_target[target_item_name]
    except KeyError:
        logger.error(f"FATAL: Target {class_name_for_log} ('{target_item_name}') not found in its vocabulary. Available keys: {list(current_vocab_map_for_target.keys())[:20]}...")
        return
    
    logger.info(f"Targeting CAM for: {class_name_for_log} (index {target_class_index}) using {args.cam_method} on {args.grad_cam_logit_target} logits.")

    if args.grad_cam_target_item and args.grad_cam_token_index is None and \
       (args.grad_cam_logit_target == "final" or args.grad_cam_logit_target == "base"): # Auto-detect for chars
        if raw_logits_for_char_search is not None and raw_logits_for_char_search.numel() > 0:
            logger.info(f"Attempting to find token index for target char '{target_item_name}' in {args.grad_cam_logit_target} logits.")
            predicted_indices_per_step = torch.argmax(raw_logits_for_char_search, dim=-1).squeeze(0)
            found_token_indices = (predicted_indices_per_step == target_class_index).nonzero(as_tuple=True)[0]
            if found_token_indices.numel() > 0:
                actual_grad_cam_token_index = found_token_indices[0].item()
                logger.info(f"Found target char '{target_item_name}' at token index {actual_grad_cam_token_index} in raw {args.grad_cam_logit_target} logits.")
                focused_item_info = f"(focused on '{target_item_name}'@tok{actual_grad_cam_token_index})"
            else:
                logger.warning(f"Target char '{target_item_name}' not found in raw {args.grad_cam_logit_target} logits. Aggregate CAM for sequence.")
                focused_item_info = f"(aggregate CAM for '{target_item_name}')"
        else:
            logger.warning(f"Raw logits for char search (type: {args.grad_cam_logit_target}) are empty. Aggregate CAM.")
            focused_item_info = f"(aggregate CAM for '{target_item_name}')"
    elif args.grad_cam_token_index is not None:
        focused_item_info = f"(explicit token {args.grad_cam_token_index} for '{target_item_name}')"
    else: 
        focused_item_info = f"(aggregate CAM for '{target_item_name}')"

    # --- Define Candidate Layers ---
    # (same as previous, ensure vision_encoder.config is valid)
    candidate_layers = []
    if hasattr(model.vision_encoder, 'config') and model.vision_encoder.config and hasattr(model.vision_encoder.config, 'num_hidden_layers'):
        ve_num_layers = model.vision_encoder.config.num_hidden_layers
        candidate_layers.extend([
            f"vision_encoder.encoder.layer.{max(0, ve_num_layers // 3 -1)}",
            f"vision_encoder.encoder.layer.{max(0, ve_num_layers // 2 -1)}",
            f"vision_encoder.encoder.layer.{ve_num_layers -1}",
        ])
    candidate_layers.extend(["base_classifier", "diacritic_classifier", "final_classifier", "diacritic_condition_proj", "shared_layer"])
    if hasattr(model, 'fusion_projection') and model.fusion_projection is not None: candidate_layers.append("fusion_projection")
    if hasattr(model, 'dynamic_fusion') and model.dynamic_fusion is not None: candidate_layers.append("dynamic_fusion")
    if hasattr(model, 'feature_enhancer') and model.feature_enhancer is not None and not isinstance(model.feature_enhancer, nn.Identity): candidate_layers.append("feature_enhancer")
    if hasattr(model, 'pos_encoder') and not isinstance(model.pos_encoder, nn.Identity): candidate_layers.append("pos_encoder")
    if hasattr(model.config, 'num_transformer_encoder_layers') and model.config.num_transformer_encoder_layers > 0:
        tf_num_layers = model.config.num_transformer_encoder_layers
        tf_layers_to_viz = sorted(list(set([0, tf_num_layers // 2, tf_num_layers - 1])))
        for i in tf_layers_to_viz:
            if i < tf_num_layers : candidate_layers.append(f"transformer_encoder.layers.{i}")
    candidate_layers = sorted(list(set(candidate_layers)))
    valid_candidate_layers = []
    for layer_name in candidate_layers:
        try: # Validate layer exists and is a module
            current_obj = model
            for part in layer_name.split('.'):
                if part.isdigit(): current_obj = current_obj[int(part)]
                else: current_obj = getattr(current_obj, part)
            if current_obj is not None and isinstance(current_obj, nn.Module) and not isinstance(current_obj, nn.Identity):
                valid_candidate_layers.append(layer_name)
        except: pass
    logger.info(f"Will attempt CAM for layers: {valid_candidate_layers}")


    generated_cam_paths = []
    cam_titles = []
    
    base_output_filename = os.path.splitext(os.path.basename(args.output_path_prefix))[0]
    # More specific output directory
    token_info_for_dir = f"_tok{actual_grad_cam_token_index}" if actual_grad_cam_token_index is not None else "_agg"
    output_dir_for_layers = os.path.join(os.path.dirname(args.output_path_prefix), 
                                         f"{base_output_filename}_{args.cam_method}_{args.grad_cam_logit_target}-{target_item_name.replace(' ','_')}{token_info_for_dir}_layers")
    os.makedirs(output_dir_for_layers, exist_ok=True)

    # Select CAM generation function
    if args.cam_method == "gradcampp": cam_generator_fn = generate_grad_cam_plus_plus
    elif args.cam_method == "layercam": cam_generator_fn = generate_layer_cam
    else: cam_generator_fn = generate_grad_cam # Default to gradcam

    for i, layer_name in enumerate(valid_candidate_layers):
        logger.info(f"--- Generating {args.cam_method} for layer ({i+1}/{len(valid_candidate_layers)}): '{layer_name}' ---")
        try:
            cam_map = cam_generator_fn(
                model, original_image_pil, pixel_values.clone(),
                processor_target_size, model.config.vision_encoder_config,
                target_class_index=target_class_index, 
                target_token_index=actual_grad_cam_token_index,
                target_layer_name=layer_name,
                grad_cam_logit_target_type=args.grad_cam_logit_target
            )
            if cam_map is not None:
                layer_name_safe = layer_name.replace('.', '_')
                item_s_file = target_item_name.replace(' ','_').replace('/','-') # Sanitize for filename
                
                individual_cam_path = os.path.join(output_dir_for_layers,
                                                   f"{args.cam_method}_{args.grad_cam_logit_target}-{item_s_file}_{layer_name_safe}{token_info_for_dir}.png")
                save_grad_cam_visualization(
                    original_image_pil, cam_map, individual_cam_path, args.grad_cam_alpha
                )
                generated_cam_paths.append(individual_cam_path)
                cam_titles.append(f"L: {layer_name}")
            else: logger.warning(f"{args.cam_method} map generation failed for layer '{layer_name}'.")
        except Exception as e_layer_cam: logger.error(f"Error generating {args.cam_method} for layer '{layer_name}': {e_layer_cam}", exc_info=True)
    
    if generated_cam_paths:
        item_s_title = target_item_name.replace(' ','_')
        combined_output_path = os.path.join(os.path.dirname(args.output_path_prefix), f"{base_output_filename}_{args.cam_method}_{args.grad_cam_logit_target}-{item_s_title}{token_info_for_dir}_all-layers.png")
        main_plot_title = (f"{args.cam_method.upper()} for {class_name_for_log} {focused_item_info}\n"
                           f"Image: {os.path.basename(args.image_path)} (Pred: '{predicted_text}')")
        combine_grad_cam_images(generated_cam_paths, combined_output_path, cam_titles, main_plot_title, args.font_path)
    else: logger.info("No CAM images were generated to combine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CAM for multiple layers for a single image.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--combined_char_vocab_path", type=str, required=True)
    parser.add_argument("--base_char_vocab_path", type=str, default="/root/hwt/VNHWT-Hierachical-Transformer-CTC/outputs/dynamic_fusion_large/base_char_vocab.json", help="Path to BASE char vocab JSON")
    parser.add_argument("--diacritic_vocab_path", type=str, default="/root/hwt/VNHWT-Hierachical-Transformer-CTC/outputs/dynamic_fusion_large/diarictic_vocab.json", help="Path to DIACRITIC vocab JSON")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path_prefix", type=str, required=True)
    
    parser.add_argument("--cam_method", type=str, default="gradcam", choices=["gradcam", "gradcampp", "layercam"], help="CAM method.")
    parser.add_argument("--grad_cam_logit_target", type=str, default="diacritic", choices=["final", "diacritic", "base"], help="Logits to target.")
    
    parser.add_argument("--grad_cam_target_item", type=str, required=True, 
                        help="Target item name (e.g., 'ò' for 'final' logits, 'o' for 'base', 'grave' for 'diacritic').")
    
    parser.add_argument("--grad_cam_token_index", type=int, default=None, help="Optional: Explicit token index for CAM.")
    parser.add_argument("--grad_cam_alpha", type=float, default=0.6)
    parser.add_argument("--font_path", type=str, default=None, help="Optional .ttf font file for plot titles.")
        
    args = parser.parse_args()

    # Validate vocab paths based on logit_target
    if args.grad_cam_logit_target == "base" and not args.base_char_vocab_path:
        parser.error("--base_char_vocab_path is required when --grad_cam_logit_target is 'base'")
    if args.grad_cam_logit_target == "diacritic" and not args.diacritic_vocab_path:
        parser.error("--diacritic_vocab_path is required when --grad_cam_logit_target is 'diacritic'")
        
    main(args)