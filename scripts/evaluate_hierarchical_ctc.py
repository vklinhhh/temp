# scripts/evaluate_hierarchical_ctc.py
import os
import sys
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
import logging
import pandas as pd
from tqdm.auto import tqdm
import json
# import itertools # Not used in current state
import numpy as np
import matplotlib.pyplot as plt # Used by evaluation_reporter indirectly
# import seaborn as sns # Used by evaluation_reporter indirectly
from collections import defaultdict

# --- Imports from project ---
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from data.ctc_ocr_dataset import CtcOcrDataset
from data.ctc_collation import ctc_collate_fn
from utils.ctc_utils import CTCDecoder
from utils.evaluation_reporter import (
    calculate_corpus_metrics, analyze_errors, generate_visualizations, create_html_report
)
from torch.utils.data import DataLoader
# from torch.nn import CTCLoss # Replaced with direct instantiation

# --- NEW: Import for visualization ---
import cv2 # For image processing in attention visualization
from PIL import Image # For image processing in attention visualization


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('evaluate_hierarchical_ctc.log')])
logger = logging.getLogger('EvaluateHierarchicalCtcScript')


# --- Helper Functions for Diacritic Analysis (remain the same, copied from your provided code) ---
def extract_diacritics(text):
    diacritics_map = {}
    diacritics = {
        'á': ('a', 'acute'), 'é': ('e', 'acute'), 'í': ('i', 'acute'), 
        'ó': ('o', 'acute'), 'ú': ('u', 'acute'), 'ý': ('y', 'acute'),
        'Á': ('A', 'acute'), 'É': ('E', 'acute'), 'Í': ('I', 'acute'),
        'Ó': ('O', 'acute'), 'Ú': ('U', 'acute'), 'Ý': ('Y', 'acute'),
        'à': ('a', 'grave'), 'è': ('e', 'grave'), 'ì': ('i', 'grave'),
        'ò': ('o', 'grave'), 'ù': ('u', 'grave'), 'ỳ': ('y', 'grave'),
        'À': ('A', 'grave'), 'È': ('E', 'grave'), 'Ì': ('I', 'grave'),
        'Ò': ('O', 'grave'), 'Ù': ('U', 'grave'), 'Ỳ': ('Y', 'grave'),
        'ả': ('a', 'hook'), 'ẻ': ('e', 'hook'), 'ỉ': ('i', 'hook'),
        'ỏ': ('o', 'hook'), 'ủ': ('u', 'hook'), 'ỷ': ('y', 'hook'),
        'Ả': ('A', 'hook'), 'Ẻ': ('E', 'hook'), 'Ỉ': ('I', 'hook'),
        'Ỏ': ('O', 'hook'), 'Ủ': ('U', 'hook'), 'Ỷ': ('Y', 'hook'),
        'ã': ('a', 'tilde'), 'ẽ': ('e', 'tilde'), 'ĩ': ('i', 'tilde'),
        'õ': ('o', 'tilde'), 'ũ': ('u', 'tilde'), 'ỹ': ('y', 'tilde'),
        'Ã': ('A', 'tilde'), 'Ẽ': ('E', 'tilde'), 'Ĩ': ('I', 'tilde'),
        'Õ': ('O', 'tilde'), 'Ũ': ('U', 'tilde'), 'Ỹ': ('Y', 'tilde'),
        'ạ': ('a', 'dot'), 'ẹ': ('e', 'dot'), 'ị': ('i', 'dot'),
        'ọ': ('o', 'dot'), 'ụ': ('u', 'dot'), 'ỵ': ('y', 'dot'),
        'Ạ': ('A', 'dot'), 'Ẹ': ('E', 'dot'), 'Ị': ('I', 'dot'),
        'Ọ': ('O', 'dot'), 'Ụ': ('U', 'dot'), 'Ỵ': ('Y', 'dot'),
        'ấ': ('a', 'circumflex_acute'), 'ầ': ('a', 'circumflex_grave'), 
        'ẩ': ('a', 'circumflex_hook'), 'ẫ': ('a', 'circumflex_tilde'), 
        'ậ': ('a', 'circumflex_dot'),
        'Ấ': ('A', 'circumflex_acute'), 'Ầ': ('A', 'circumflex_grave'), 
        'Ẩ': ('A', 'circumflex_hook'), 'Ẫ': ('A', 'circumflex_tilde'), 
        'Ậ': ('A', 'circumflex_dot'),
        'ế': ('e', 'circumflex_acute'), 'ề': ('e', 'circumflex_grave'), 
        'ể': ('e', 'circumflex_hook'), 'ễ': ('e', 'circumflex_tilde'), 
        'ệ': ('e', 'circumflex_dot'),
        'Ế': ('E', 'circumflex_acute'), 'Ề': ('E', 'circumflex_grave'), 
        'Ể': ('E', 'circumflex_hook'), 'Ễ': ('E', 'circumflex_tilde'), 
        'Ệ': ('E', 'circumflex_dot'),
        'ố': ('o', 'circumflex_acute'), 'ồ': ('o', 'circumflex_grave'), 
        'ổ': ('o', 'circumflex_hook'), 'ỗ': ('o', 'circumflex_tilde'), 
        'ộ': ('o', 'circumflex_dot'),
        'Ố': ('O', 'circumflex_acute'), 'Ồ': ('O', 'circumflex_grave'), 
        'Ổ': ('O', 'circumflex_hook'), 'Ỗ': ('O', 'circumflex_tilde'), 
        'Ộ': ('O', 'circumflex_dot'),
        'ắ': ('a', 'breve_acute'), 'ằ': ('a', 'breve_grave'), 
        'ẳ': ('a', 'breve_hook'), 'ẵ': ('a', 'breve_tilde'), 
        'ặ': ('a', 'breve_dot'),
        'Ắ': ('A', 'breve_acute'), 'Ằ': ('A', 'breve_grave'), 
        'Ẳ': ('A', 'breve_hook'), 'Ẵ': ('A', 'breve_tilde'), 
        'Ặ': ('A', 'breve_dot'),
        'ớ': ('o', 'horn_acute'), 'ờ': ('o', 'horn_grave'), 
        'ở': ('o', 'horn_hook'), 'ỡ': ('o', 'horn_tilde'), 
        'ợ': ('o', 'horn_dot'),
        'Ớ': ('O', 'horn_acute'), 'Ờ': ('O', 'horn_grave'), 
        'Ở': ('O', 'horn_hook'), 'Ỡ': ('O', 'horn_tilde'), 
        'Ợ': ('O', 'horn_dot'),
        'ứ': ('u', 'horn_acute'), 'ừ': ('u', 'horn_grave'), 
        'ử': ('u', 'horn_hook'), 'ữ': ('u', 'horn_tilde'), 
        'ự': ('u', 'horn_dot'),
        'Ứ': ('U', 'horn_acute'), 'Ừ': ('U', 'horn_grave'), 
        'Ử': ('U', 'horn_hook'), 'Ữ': ('U', 'horn_tilde'), 
        'Ự': ('U', 'horn_dot'),
        'â': ('a', 'circumflex'), 'ê': ('e', 'circumflex'), 'ô': ('o', 'circumflex'),
        'Â': ('A', 'circumflex'), 'Ê': ('E', 'circumflex'), 'Ô': ('O', 'circumflex'),
        'ă': ('a', 'breve'), 'Ă': ('A', 'breve'),
        'ơ': ('o', 'horn'), 'ư': ('u', 'horn'),
        'Ơ': ('O', 'horn'), 'Ư': ('U', 'horn'),
        'đ': ('d', 'stroke'), 'Đ': ('D', 'stroke'),
    }
    for i, char in enumerate(text):
        if char in diacritics:
            base_char, diacritic = diacritics[char]
            diacritics_map[i] = {'char': char, 'base': base_char, 'diacritic': diacritic}
    return diacritics_map

def analyze_diacritic_accuracy(ground_truths, predictions):
    diacritic_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    for gt, pred in zip(ground_truths, predictions):
        if len(gt) != len(pred): continue
        gt_diacritics = extract_diacritics(gt)
        pred_diacritics = extract_diacritics(pred)
        for i in gt_diacritics:
            if i < len(pred):
                gt_diac = gt_diacritics[i]['diacritic']
                diacritic_stats[gt_diac]['total'] += 1
                if i in pred_diacritics and pred_diacritics[i]['diacritic'] == gt_diac:
                    diacritic_stats[gt_diac]['correct'] += 1
                else:
                    error_example = {
                        'gt_char': gt_diacritics[i]['char'],
                        'pred_char': pred[i] if i < len(pred) else '',
                        'gt_context': gt[max(0, i-5):min(len(gt), i+6)],
                        'pred_context': pred[max(0, i-5):min(len(pred), i+6)],
                    }
                    if len(diacritic_stats[gt_diac]['examples']) < 10:
                        diacritic_stats[gt_diac]['examples'].append(error_example)
    results = {}
    for diac, stats in diacritic_stats.items():
        if stats['total'] > 0:
            results[diac] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
                'examples': stats['examples']
            }
    return results

def visualize_diacritic_accuracy(diacritic_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    diacritics, accuracies, counts = [], [], []
    for diac, stats in diacritic_results.items():
        diacritics.append(diac); accuracies.append(stats['accuracy'] * 100); counts.append(stats['total'])
    
    if not diacritics: # Handle case with no diacritic results
        logger.warning("No diacritic results to visualize.")
        return None

    sorted_indices = np.argsort(counts)[::-1]
    diacritics = [diacritics[i] for i in sorted_indices]; accuracies = [accuracies[i] for i in sorted_indices]; counts = [counts[i] for i in sorted_indices]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(diacritics, accuracies, color='skyblue')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count), ha='center', va='bottom', rotation=0, fontsize=9, color='black')
    mean_acc_val = np.mean(accuracies) if accuracies else 0.0
    plt.axhline(y=mean_acc_val, color='r', linestyle='--', label=f'Average: {mean_acc_val:.1f}%')
    plt.xlabel('Diacritic Type'); plt.ylabel('Accuracy (%)'); plt.title('Diacritic Recognition Accuracy by Type')
    plt.xticks(rotation=45, ha='right'); plt.ylim(0, 105); plt.tight_layout(); plt.legend(); plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, 'diacritic_accuracy.png')
    plt.savefig(save_path); plt.close()
    return save_path


# --- NEW: Helper function for attention visualization ---
def save_attention_visualization(
    original_pil_image,
    attention_map_1d, 
    vision_encoder_config,
    processor_target_size, 
    output_path_prefix,
    sample_idx,
    ground_truth_text,
    predicted_text
):
    try:
        if isinstance(attention_map_1d, torch.Tensor):
            attention_map_1d = attention_map_1d.cpu().numpy()

        patch_size = vision_encoder_config.patch_size
        processed_height, processed_width = processor_target_size
        num_patches_h = processed_height // patch_size
        num_patches_w = processed_width // patch_size
        
        expected_num_patches = num_patches_h * num_patches_w
        if attention_map_1d.shape[0] == expected_num_patches + 1: # Has CLS token
            attention_map_patches = attention_map_1d[1:, :] 
        elif attention_map_1d.shape[0] == expected_num_patches:
            attention_map_patches = attention_map_1d
        else:
            logger.warning(f"Sample {sample_idx}: Mismatch in attention map length ({attention_map_1d.shape[0]}) vs expected ({expected_num_patches} or {expected_num_patches+1}). Skipping viz.")
            return None

        original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        img_h, img_w = original_cv_image.shape[:2]
        
        region_names = ["Above", "Middle", "Below"]
        saved_files = []

        for i in range(attention_map_patches.shape[1]): # Iterate through the 3 regions
            region_attention_1d = attention_map_patches[:, i]
            try:
                attention_grid_2d = region_attention_1d.reshape(num_patches_h, num_patches_w)
            except ValueError as reshape_err:
                logger.error(f"Sample {sample_idx}: Error reshaping attention for region {region_names[i]}: {reshape_err}. Skipping region.")
                continue

            heatmap_resized_raw = cv2.resize(attention_grid_2d, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            heatmap_normalized = cv2.normalize(heatmap_resized_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Convert explicitly to uint8 *after* normalization if needed for applyColorMap
            heatmap_normalized_uint8 = heatmap_normalized.astype(np.uint8) 
            heatmap_color = cv2.applyColorMap(heatmap_normalized_uint8, cv2.COLORMAP_JET)
            overlayed_image = cv2.addWeighted(original_cv_image, 0.6, heatmap_color, 0.4, 0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlayed_image, f"GT: {ground_truth_text}", (10, 30), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(overlayed_image, f"Pred: {predicted_text}", (10, 60), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(overlayed_image, f"Region: {region_names[i]}", (10, img_h - 20), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

            output_filename = f"{output_path_prefix}_sample{sample_idx}_region{i}_{region_names[i]}.png"
            cv2.imwrite(output_filename, overlayed_image)
            saved_files.append(output_filename)
            logger.debug(f"Saved attention visualization to {output_filename}") # Changed to debug for less verbose default
        
        return saved_files
    except Exception as e:
        logger.error(f"Error generating attention visualization for sample {sample_idx}: {e}", exc_info=True)
        return None


def run_hierarchical_evaluation(
    model_path,
    dataset_name,
    output_dir,
    combined_char_vocab_path,
    dataset_split='test',
    batch_size=16,
    num_workers=4,
    device=None,
    num_attention_visualizations=5 # NEW: Control number of attention visualizations
    ):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info(f"Loading combined char vocab from: {combined_char_vocab_path}")
        with open(combined_char_vocab_path, 'r', encoding='utf-8') as f: combined_char_vocab = json.load(f)
        if not combined_char_vocab: raise ValueError("Combined vocab empty.")
        combined_char_to_idx = {c: i for i, c in enumerate(combined_char_vocab)}
        combined_idx_to_char = {i: c for i, c in enumerate(combined_char_vocab)}
        blank_idx = combined_char_to_idx.get('<blank>', 0)
        logger.info(f"Combined vocab loaded: {len(combined_char_vocab)} chars.")
    except Exception as e: logger.error(f"FATAL: Vocab load fail: {e}", exc_info=True); return

    try:
        logger.info(f"Loading trained Hierarchical CTC model from: {model_path}")
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_path, combined_char_vocab=combined_char_vocab)
        processor = model.processor
        model.to(device); model.eval()
        logger.info("Model and processor loaded.")
        # Log component status
        for component_flag in ['use_visual_diacritic_attention', 'use_character_diacritic_compatibility', 'use_few_shot_diacritic_adapter', 'use_dynamic_fusion', 'use_feature_enhancer']:
             if hasattr(model.config, component_flag):
                 logger.info(f"Model config '{component_flag}': {getattr(model.config, component_flag)}")
    except Exception as e: logger.error(f"FATAL: Failed to load model: {e}", exc_info=True); return

    # --- Determine processor target size for attention visualization ---
    processor_target_size = None
    if hasattr(processor, 'size') and isinstance(processor.size, dict) and "height" in processor.size and "width" in processor.size:
        processor_target_size = (processor.size["height"], processor.size["width"])
    elif hasattr(model.config, 'vision_encoder_config') and hasattr(model.config.vision_encoder_config, 'image_size'):
        img_s = model.config.vision_encoder_config.image_size
        if isinstance(img_s, int): processor_target_size = (img_s, img_s)
    
    if processor_target_size is None:
        logger.warning("Could not determine processor target size. Using default (384, 384) for attention visualization. THIS MAY BE INCORRECT.")
        processor_target_size = (384, 384) 
    else:
        logger.info(f"Using processor target size for attention viz: {processor_target_size}")

    try:
        logger.info(f"Loading test dataset: {dataset_name}, split: {dataset_split}")
        hf_dataset = load_dataset(dataset_name)
        current_split = dataset_split
        if dataset_split not in hf_dataset:
             if 'test' in hf_dataset: current_split = 'test'
             elif 'validation' in hf_dataset: current_split = 'validation'
             else: raise ValueError(f"Split '{dataset_split}' not found. Available: {list(hf_dataset.keys())}")
             logger.warning(f"Using dataset split: '{current_split}' as '{dataset_split}' was not found.")
        test_hf_split = hf_dataset[current_split]
        logger.info(f"Test set size: {len(test_hf_split)}")
        if not any(col in test_hf_split.column_names for col in ['label', 'word', 'text']):
             raise ValueError("Dataset needs 'label', 'word', or 'text' column.")
    except Exception as dataset_load_e: logger.error(f"FATAL: Dataset load failed: {dataset_load_e}", exc_info=True); return

    try:
        logger.info("Creating CTC test dataset wrapper (using combined vocab)...")
        # Pass is_training=False so CtcOcrDataset returns 'original_image_pil'
        test_dataset = CtcOcrDataset(test_hf_split, processor, combined_char_to_idx, unk_token='[UNK]', is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        logger.info(f"Test DataLoader created.")
    except Exception as dataset_wrap_e: logger.error(f"FATAL: Dataset/loader failed: {dataset_wrap_e}", exc_info=True); return

    ctc_decoder = CTCDecoder(idx_to_char_map=combined_idx_to_char, blank_idx=blank_idx)
    all_predictions, all_ground_truths = [], []
    total_loss_eval, batch_count_eval = 0.0, 0
    ctc_loss_fn_eval = nn.CTCLoss(blank=blank_idx, reduction='sum', zero_infinity=True)

    # --- NEW: For attention visualization ---
    attention_viz_samples_collected = 0
    attention_viz_dir = os.path.join(output_dir, "attention_visualizations_eval") # Changed folder name
    os.makedirs(attention_viz_dir, exist_ok=True)
    attention_report_data = []
    global_sample_idx_tracker = 0

    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Test Set")
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: continue
            try:
                pixel_values = batch['pixel_values'].to(device)
                labels_cpu = batch['labels'] 
                label_lengths_cpu = batch['label_lengths']
                texts_gt = batch['texts']
                original_images_pil_batch = batch.get('original_images_pil') # Get from collate_fn
                current_batch_size_actual = pixel_values.size(0)

                should_return_attention = (
                    model.config.use_visual_diacritic_attention and
                    attention_viz_samples_collected < num_attention_visualizations and
                    original_images_pil_batch is not None and
                    len(original_images_pil_batch) == current_batch_size_actual # Ensure lists match
                )
                
                outputs = model(pixel_values=pixel_values, return_diacritic_attention=should_return_attention)
                logits = outputs.get('logits')
                if logits is None: logger.warning(f"Logits are None for batch {batch_idx}, skipping."); continue

                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                time_steps = log_probs.size(0)
                input_lengths_cpu = torch.full((current_batch_size_actual,), time_steps, dtype=torch.long, device='cpu')
                
                try: 
                    loss = ctc_loss_fn_eval(log_probs.cpu(), labels_cpu, input_lengths_cpu, label_lengths_cpu)
                    total_loss_eval += loss.item()
                except Exception as loss_e: logger.warning(f"Loss calc error: {loss_e}")

                decoded_preds = ctc_decoder(logits)
                all_predictions.extend(decoded_preds); all_ground_truths.extend(texts_gt)

                if should_return_attention:
                    attention_maps_batch = outputs.get('visual_diacritic_attention_maps')
                    if attention_maps_batch is not None:
                        for i in range(attention_maps_batch.shape[0]):
                            if attention_viz_samples_collected < num_attention_visualizations:
                                current_original_image = original_images_pil_batch[i]
                                current_attention_map = attention_maps_batch[i]
                                current_gt = texts_gt[i]; current_pred = decoded_preds[i]
                                
                                viz_output_prefix = os.path.join(attention_viz_dir, f"att_viz")
                                saved_files = save_attention_visualization(
                                    current_original_image, current_attention_map,
                                    model.config.vision_encoder_config, processor_target_size,
                                    viz_output_prefix, global_sample_idx_tracker + i,
                                    current_gt, current_pred
                                )
                                if saved_files:
                                    attention_report_data.append({
                                        "sample_idx": global_sample_idx_tracker + i, "gt": current_gt, "pred": current_pred,
                                        "image_paths": [os.path.relpath(p, output_dir) for p in saved_files]
                                    })
                                attention_viz_samples_collected += 1
                batch_count_eval += 1
            except Exception as eval_batch_e: logger.error(f"Error eval batch {batch_idx}: {eval_batch_e}", exc_info=True); continue
            global_sample_idx_tracker += current_batch_size_actual
    logger.info("Evaluation loop finished.")

    if not all_ground_truths: logger.error("No samples processed."); return
    df_results = pd.DataFrame({'GroundTruth': all_ground_truths, 'Prediction': all_predictions})
    results_csv_path = os.path.join(output_dir, "evaluation_results_raw.csv")
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"Saved detailed results to: {results_csv_path}")
    
    num_samples_eval = len(df_results)
    avg_loss = total_loss_eval / num_samples_eval if num_samples_eval > 0 else 0.0
    cer, wer = calculate_corpus_metrics(df_results['Prediction'].tolist(), df_results['GroundTruth'].tolist())
    logger.info(f"Final Evaluation Metrics: Avg Loss={avg_loss:.4f}, CER={cer:.4f}, WER={wer:.4f}")
    
    logger.info("Analyzing diacritic recognition accuracy...")
    diacritic_results = analyze_diacritic_accuracy(df_results['GroundTruth'].tolist(), df_results['Prediction'].tolist())
    
    std_viz_dir = os.path.join(output_dir, "visualizations") # Standard visualizations like length distribution
    diacritic_viz_path = visualize_diacritic_accuracy(diacritic_results, std_viz_dir) # Save diac acc plot here
    
    logger.info("Diacritic Recognition Accuracy:")
    for diac, stats in sorted(diacritic_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"  {diac}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    diacritic_json_path = os.path.join(output_dir, "diacritic_analysis.json")
    with open(diacritic_json_path, 'w', encoding='utf-8') as f: json.dump(diacritic_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved diacritic analysis to: {diacritic_json_path}")
    
    summary_stats = {'model_path': model_path, 'dataset_name': dataset_name, 'dataset_split': current_split, 'total_samples': num_samples_eval, 'avg_loss': avg_loss, 'cer': cer, 'wer': wer}
    if diacritic_results: summary_stats['diacritic_avg_accuracy'] = np.mean([stats['accuracy'] for stats in diacritic_results.values() if stats['total'] > 0])
    else: summary_stats['diacritic_avg_accuracy'] = 0.0
    
    if hasattr(model.config, 'use_dynamic_fusion'): summary_stats['use_dynamic_fusion'] = model.config.use_dynamic_fusion
    if hasattr(model.config, 'use_feature_enhancer'): summary_stats['use_feature_enhancer'] = model.config.use_feature_enhancer
    
    error_analysis = analyze_errors(df_results)
    _ = generate_visualizations(df_results, output_dir) # Standard visualizations saved in output_dir/visualizations
    
    report_path = os.path.join(output_dir, "evaluation_report.html")
    # Pass std_viz_dir for standard plots, and attention_report_data for attention plots
    create_html_report(report_path, summary_stats, error_analysis, std_viz_dir, attention_report_data) 
    
    diacritic_report_path = os.path.join(output_dir, "diacritic_evaluation_report.html")
    # Pass the path to the specific diacritic accuracy plot
    create_diacritic_html_report(diacritic_report_path, diacritic_results, model_path, diacritic_viz_path if diacritic_viz_path else "")
    
    logger.info(f"Evaluation complete. Reports: {report_path}, {diacritic_report_path}")


# --- create_diacritic_html_report (remains the same, copied from your provided code) ---
def create_diacritic_html_report(report_path, diacritic_results, model_path, viz_path):
    logger.info(f"Generating diacritic HTML report at: {report_path}")
    total_correct = sum(stats['correct'] for stats in diacritic_results.values())
    total_diacritics = sum(stats['total'] for stats in diacritic_results.values())
    overall_accuracy = total_correct / total_diacritics if total_diacritics > 0 else 0
    sorted_diacritics = sorted(diacritic_results.items(), key=lambda x: (x[1]['accuracy'], -x[1]['total']), reverse=True)
    html = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Hierarchical CTC OCR: Diacritic Evaluation Report</title>
<style>body{{font-family:sans-serif;margin:20px}}h1,h2,h3{{color:#333}}table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background-color:#f2f2f2}}
.summary-box{{border:1px solid #ccc;padding:15px;margin-bottom:20px;background-color:#f9f9f9}}
.metrics td{{font-weight:bold;color:#0056b3}}.high{{color:green}}.medium{{color:#cc7000}}.low{{color:red}}
img{{max-width:800px;height:auto;display:block;margin:20px auto;border:1px solid #ccc}}
.examples{{font-family:monospace;background-color:#f5f5f5;padding:10px;margin:5px 0;border-radius:4px}}</style></head><body>
<h1>Hierarchical CTC OCR: Diacritic Evaluation Report</h1><p>Model Path: {model_path}</p>
<h2>Summary</h2><div class="summary-box metrics"><table>
<tr><td>Total Diacritics Evaluated</td><td>{total_diacritics}</td></tr>
<tr><td>Overall Diacritic Accuracy</td><td>{overall_accuracy:.4f} ({total_correct}/{total_diacritics})</td></tr>
<tr><td>Number of Diacritic Types</td><td>{len(diacritic_results)}</td></tr></table></div>"""
    if viz_path and os.path.exists(viz_path):
        html += f"""<h2>Diacritic Accuracy Visualization</h2>
                   <img src="{os.path.relpath(viz_path, os.path.dirname(report_path))}" alt="Diacritic Accuracy Chart">"""
    else:
        html += "<h2>Diacritic Accuracy Visualization</h2><p>Visualization not available.</p>"

    html += """<h2>Diacritic Type Breakdown</h2><table><tr><th>Diacritic</th><th>Accuracy</th><th>Correct</th><th>Total</th><th>Example Errors</th></tr>"""
    for diac, stats in sorted_diacritics:
        accuracy = stats['accuracy']
        accuracy_class = 'high' if accuracy >= 0.9 else ('medium' if accuracy >= 0.7 else 'low')
        example_html = ""
        if stats['examples']:
            for i, example in enumerate(stats['examples'][:3]):
                example_html += f"""<div class="examples"><strong>Error {i+1}:</strong><br>GT: {example['gt_char']} (in "{example['gt_context']}")<br>Pred: {example['pred_char']} (in "{example['pred_context']}")</div>"""
        else: example_html = "<em>No errors found</em>"
        html += f"""<tr><td>{diac}</td><td class="{accuracy_class}">{accuracy:.4f} ({accuracy*100:.1f}%)</td><td>{stats['correct']}</td><td>{stats['total']}</td><td>{example_html}</td></tr>"""
    html += """</table><h2>Analysis and Recommendations</h2><div class="summary-box"><h3>Diacritic Categories Performance</h3>"""
    categories = {'tone_marks':['acute','grave','hook','tilde','dot'],'modifiers':['circumflex','breve','horn','stroke'],'combined':[d for d in diacritic_results.keys() if '_' in d]}
    category_stats = {}
    for category, diacritics_list in categories.items(): # Corrected variable name
        category_diacritics = [d for d in diacritics_list if d in diacritic_results]
        if category_diacritics:
            correct = sum(diacritic_results[d]['correct'] for d in category_diacritics)
            total = sum(diacritic_results[d]['total'] for d in category_diacritics)
            accuracy = correct / total if total > 0 else 0
            category_stats[category] = {'accuracy': accuracy, 'correct': correct, 'total': total}
    for category, stats in category_stats.items():
        category_display = category.replace('_',' ').title()
        accuracy_class = 'high' if stats['accuracy'] >= 0.9 else ('medium' if stats['accuracy'] >= 0.7 else 'low')
        html += f"""<p><strong>{category_display}:</strong> <span class="{accuracy_class}">{stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})</span></p>"""
    html += """<h3>Key Findings and Recommendations</h3><ul>"""
    problem_diacritics = [(d, stats) for d, stats in diacritic_results.items() if stats['accuracy'] < 0.8 and stats['total'] >= 10]
    if problem_diacritics:
        html += "<li><strong>Problematic Diacritics:</strong><ul>"
        for diac, stats in sorted(problem_diacritics, key=lambda x: x[1]['accuracy']):
            html += f"""<li><strong>{diac}</strong>: Accuracy {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})<br>Recommendation: Consider enhancing the model's ability to recognize this diacritic.</li>"""
        html += "</ul></li>"
    simple_diacritics = [d for d in diacritic_results.keys() if '_' not in d and d != 'no_diacritic']
    combined_diacritics_list = [d for d in diacritic_results.keys() if '_' in d] # Corrected variable name
    if simple_diacritics and combined_diacritics_list:
        simple_acc = sum(diacritic_results[d]['correct'] for d in simple_diacritics) / sum(diacritic_results[d]['total'] for d in simple_diacritics) if sum(diacritic_results[d]['total'] for d in simple_diacritics) > 0 else 0
        combined_acc = sum(diacritic_results[d]['correct'] for d in combined_diacritics_list) / sum(diacritic_results[d]['total'] for d in combined_diacritics_list) if sum(diacritic_results[d]['total'] for d in combined_diacritics_list) > 0 else 0
        if combined_acc < simple_acc - 0.1:
            html += f"""<li><strong>Combined vs. Simple Diacritics:</strong> Combined diacritics have significantly lower accuracy ({combined_acc:.4f}) than simple diacritics ({simple_acc:.4f}).<br>Recommendation: Consider enhancing the model's capability with combined diacritics, potentially through:<ul><li>Adding more training samples with combined diacritics</li><li>Using the Few-Shot Diacritic Adapter for rare combinations</li><li>Fine-tuning the Visual Diacritic Attention mechanism</li><li>Enabling Dynamic Fusion to better capture multi-scale features</li><li>Enabling Feature Enhancer for better local feature detection</li></ul></li>"""
    overall_recommendation = ""
    if overall_accuracy < 0.7: overall_recommendation = """<li><strong>General Performance:</strong> Overall diacritic accuracy is low.<br>Recommendation: Consider fundamental changes to the diacritic classifier, such as:<ul><li>Enable all three diacritic enhancement modules</li><li>Enable Dynamic Multi-Scale Fusion and Feature Enhancer</li><li>Increase training data with diverse diacritical marks</li><li>Consider a specialized training phase focusing only on diacritics</li></ul></li>"""
    elif overall_accuracy < 0.9: overall_recommendation = """<li><strong>General Performance:</strong> Overall diacritic accuracy is moderate.<br>Recommendation: Some targeted improvements could help:<ul><li>Enable Visual Diacritic Attention to better focus on diacritic regions</li><li>Add more examples of the lower-performing diacritics to the training set</li><li>Use Character-Diacritic Compatibility to improve linguistic consistency</li><li>Consider enabling Dynamic Fusion for better multi-scale feature extraction</li></ul></li>"""
    else: overall_recommendation = """<li><strong>General Performance:</strong> Overall diacritic accuracy is excellent.<br>Recommendation: Focus on the few problematic diacritics if any.</li>"""
    html += overall_recommendation + """</ul></div></body></html>"""
    try:
        with open(report_path, 'w', encoding='utf-8') as f: f.write(html)
        logger.info("Diacritic HTML report generated successfully.")
    except Exception as e: logger.error(f"Failed to write diacritic HTML report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hierarchical CTC OCR model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained Hierarchical CTC model directory")
    parser.add_argument("--combined_char_vocab_path", type=str, required=True, help="Path to COMBINED char vocab JSON")
    parser.add_argument("--dataset_name", type=str, required=True, help="Test dataset name/path.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_hierarchical_ctc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_attention_visualizations", type=int, default=5, help="Number of samples to generate attention visualizations for.") # NEW CLI arg
    args = parser.parse_args()

    run_hierarchical_evaluation(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        combined_char_vocab_path=args.combined_char_vocab_path,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_attention_visualizations=args.num_attention_visualizations # Pass to function
    )