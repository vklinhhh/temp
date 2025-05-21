# scripts/compare_compatibility_impact.py
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from jiwer import cer as calculate_cer, wer as calculate_wer # For CER/WER calculation

# Adjust imports based on your project structure
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from data.ctc_ocr_dataset import CtcOcrDataset # We'll use its processing logic
from data.ctc_collation import ctc_collate_fn # For dataloader
from utils.ctc_utils import (
    build_combined_vietnamese_charset, # For default vocab generation
    decompose_vietnamese_char, # CRUCIAL for diacritic error analysis
    get_char_type # Helper for diacritic analysis
)
from utils.compatibility_logging import visualize_compatibility_matrix # For visualizing matrix

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('compatibility_comparison_detailed_report.log'),
    ],
)
logger = logging.getLogger('CompareCompatibilityDetailedScript')

# --- Helper Functions from ctc_trainer (or similar decoding logic) ---
def greedy_decode_ctc(logits, blank_idx, idx_to_char):
    """Decodes CTC output logits using greedy search."""
    if logits.ndim != 3: raise ValueError(f"Logits 3D, got {logits.shape}")
    # Ensure logits are on CPU for argmax and further processing if not already
    # Detach is important if grads are not needed.
    predicted_ids = torch.argmax(logits.cpu().detach(), dim=-1) 
    decoded_batch = []

    for pred_ids_single_sample in predicted_ids: # pred_ids_single_sample is a 1D tensor for one sample
        # Get unique consecutive elements. We don't need return_counts=True for this part.
        # The first element of the tuple returned by unique_consecutive is what we need.
        unique_elements = torch.unique_consecutive(pred_ids_single_sample)
        
        # Remove blanks
        cleaned_ids = [p.item() for p in unique_elements if p.item() != blank_idx] # .item() to get Python numbers
        
        try:
            # Map cleaned IDs to characters
            decoded_elements = [idx_to_char.get(idx, '?') for idx in cleaned_ids]
            decoded_string = "".join(decoded_elements) # Typically no delimiter for greedy decode string
        except Exception as e:
            logger.warning(f"Decode map error: {e}. Sequence IDs: {cleaned_ids}")
            decoded_string = "<DECODE_ERROR>"
        decoded_batch.append(decoded_string)
    return decoded_batch

def get_base_diacritic_preds(final_preds_strings, base_char_vocab, diacritic_vocab):
    """
    Infers base and diacritic sequences from final predicted strings.
    This is an approximation for evaluating hierarchical heads.
    """
    pred_base_strings = []
    pred_diac_strings = []
    
    base_char_to_idx = {c: i for i, c in enumerate(base_char_vocab)}
    diac_to_idx = {d: i for i, d in enumerate(diacritic_vocab)}
    unk_base_idx = base_char_to_idx.get('<unk>', 0)
    unk_diac_idx = diac_to_idx.get('<unk>', 0)
    no_diac_idx = diac_to_idx.get('no_diacritic', 1)


    for final_str in final_preds_strings:
        current_base_seq = []
        current_diac_seq = []
        for char_final in final_str:
            base, diac_name, _ = decompose_vietnamese_char(char_final)
            current_base_seq.append(base if base else char_final) # Fallback to full char if no base
            current_diac_seq.append(diac_name if diac_name else 'no_diacritic')
        pred_base_strings.append("".join(current_base_seq))
        pred_diac_strings.append(" ".join(current_diac_seq)) # Diacritics as space-separated sequence
    return pred_base_strings, pred_diac_strings


def analyze_diacritic_errors(gt_strings, pred_strings):
    """Analyzes specific types of diacritic errors."""
    errors = {
        "total_chars_gt": 0,
        "total_chars_pred": 0,
        "vowels_gt": 0,
        "vowels_pred": 0,
        "diacritics_gt_count": 0,     # Number of characters in GT that have a diacritic
        "diacritics_pred_count": 0,   # Number of characters in Pred that have a diacritic
        "correctly_placed_diacritics": 0, # Base and Diacritic both match GT
        "correct_base_wrong_diacritic": 0,
        "wrong_base_correct_diacritic": 0, # Less common but possible
        "missing_diacritic_on_vowel": 0,  # GT has vowel+diac, Pred has vowel_only
        "extra_diacritic_on_vowel": 0,    # GT has vowel_only, Pred has vowel+diac
        "diacritic_on_consonant_error": 0 # Pred has consonant+diac (usually error)
    }
    # This is a complex task. For a simpler version, we compare char by char after alignment (e.g., from CER calculation tools)
    # For now, let's do a high-level count.
    
    for gt_s, pred_s in zip(gt_strings, pred_strings):
        # A more robust way would be to align the strings first (e.g., using Levenshtein alignment)
        # For simplicity here, we iterate and compare decompositions.
        
        gt_decomposed = [decompose_vietnamese_char(c) for c in gt_s]
        pred_decomposed = [decompose_vietnamese_char(c) for c in pred_s]
        errors["total_chars_gt"] += len(gt_s)
        errors["total_chars_pred"] += len(pred_s)

        for gt_base, gt_diac_name, _ in gt_decomposed:
            if get_char_type(gt_base) == 'vowel': errors["vowels_gt"] += 1
            if gt_diac_name and gt_diac_name != 'no_diacritic': errors["diacritics_gt_count"] += 1
        
        for pred_base, pred_diac_name, _ in pred_decomposed:
            if get_char_type(pred_base) == 'vowel': errors["vowels_pred"] += 1
            if pred_diac_name and pred_diac_name != 'no_diacritic':
                errors["diacritics_pred_count"] += 1
                if get_char_type(pred_base) == 'consonant':
                    errors["diacritic_on_consonant_error"] +=1
        
        # More detailed alignment needed for correct_base_wrong_diacritic etc.
        # This can be added using `editdistance` or similar to get operations.
        # For now, these counts are simpler proxies.

    return errors


def evaluate_model_extended(model, dataloader, device, idx_to_char_final, idx_to_char_base, idx_to_char_diac, blank_idx_final, blank_idx_base, blank_idx_diac, model_type_prefix=""):
    """Extended evaluation including hierarchical heads (approximate) and diacritic analysis."""
    model.eval()
    all_final_preds = []
    all_final_gts = []
    all_base_preds_approx = [] # Approximated from final_preds
    all_base_gts_approx = []   # Approximated from final_gts
    all_diac_preds_approx = [] # Approximated from final_preds
    all_diac_gts_approx = []   # Approximated from final_gts

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type_prefix}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device) # Final combined labels
            label_lengths = batch["label_lengths"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels, label_lengths=label_lengths)
            
            loss = outputs['loss']
            if loss is not None: # Loss might be None if evaluating without labels for some reason
                total_loss += loss.item() * pixel_values.size(0) # Weighted by batch size
            total_samples += pixel_values.size(0)

            # Decode final predictions
            final_logits = outputs['logits']
            pred_strings_final = greedy_decode_ctc(final_logits, blank_idx_final, idx_to_char_final)
            all_final_preds.extend(pred_strings_final)
            
            gt_strings_final = []
            for i in range(labels.size(0)):
                valid_labels = labels[i][:label_lengths[i]]
                gt_strings_final.append("".join([idx_to_char_final.get(label_id.item(), "?") for label_id in valid_labels]))
            all_final_gts.extend(gt_strings_final)

            # Approximate base/diacritic GTs and Preds from final strings
            # This is a simplification. True hierarchical evaluation would need aligned frame-level labels for base/diac.
            gt_base_approx, gt_diac_approx = get_base_diacritic_preds(gt_strings_final, list(idx_to_char_base.values()), list(idx_to_char_diac.values()))
            pred_base_approx, pred_diac_approx = get_base_diacritic_preds(pred_strings_final, list(idx_to_char_base.values()), list(idx_to_char_diac.values()))
            all_base_gts_approx.extend(gt_base_approx)
            all_base_preds_approx.extend(pred_base_approx)
            all_diac_gts_approx.extend(gt_diac_approx) # These are space-separated diacritic names
            all_diac_preds_approx.extend(pred_diac_approx)


    metrics = {}
    if total_samples > 0 and total_loss > 0: # Ensure loss was computed
      metrics[f'{model_type_prefix}eval_loss'] = total_loss / total_samples
    else:
      metrics[f'{model_type_prefix}eval_loss'] = float('nan')


    # Final OCR Metrics
    metrics[f'{model_type_prefix}eval_cer'] = calculate_cer(all_final_gts, all_final_preds)
    metrics[f'{model_type_prefix}eval_wer'] = calculate_wer(all_final_gts, all_final_preds)

    # Approximate Hierarchical Head Performance
    # CER for base characters (approximated)
    metrics[f'{model_type_prefix}eval_base_cer_approx'] = calculate_cer(all_base_gts_approx, all_base_preds_approx)
    # For diacritics, WER might be more suitable as they are sequences of diacritic names
    metrics[f'{model_type_prefix}eval_diac_wer_approx'] = calculate_wer(all_diac_gts_approx, all_diac_preds_approx)

    # Diacritic Error Analysis
    diac_errors = analyze_diacritic_errors(all_final_gts, all_final_preds)
    for k, v in diac_errors.items():
        metrics[f'{model_type_prefix}diac_err_{k}'] = v
        
    # Store some example predictions
    metrics[f'{model_type_prefix}example_gts'] = all_final_gts[:5]
    metrics[f'{model_type_prefix}example_preds'] = all_final_preds[:5]

    return metrics, all_final_gts, all_final_preds


def load_model_for_eval(model_path, device):
    logger.info(f"Loading model from: {model_path}")
    if not os.path.isdir(model_path):
        logger.error(f"Model path {model_path} is not a directory.")
        return None, None
    try:
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        processor = model.processor
        if processor is None:
             from transformers import AutoProcessor
             try: processor = AutoProcessor.from_pretrained(model_path)
             except Exception: processor = AutoProcessor.from_pretrained(model.config.vision_encoder_name, trust_remote_code=True)
        logger.info(f"Model and processor loaded successfully from {model_path}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None, None

def get_dataset_slice(dataset_name, start_idx, num_samples, split_name='train'):
    try:
        full_dataset = load_dataset(dataset_name, split=f"{split_name}", streaming=False) # Ensure not streaming for select
        end_idx = start_idx + num_samples
        if end_idx > len(full_dataset):
            logger.warning(f"Requested slice end ({end_idx}) exceeds dataset size ({len(full_dataset)}). Truncating.")
            end_idx = len(full_dataset)
        
        if start_idx >= len(full_dataset) or start_idx >= end_idx:
            logger.error(f"Invalid slice range: start {start_idx}, end {end_idx} for dataset size {len(full_dataset)}.")
            return None

        dataset_slice = full_dataset.select(range(start_idx, end_idx))
        logger.info(f"Selected {len(dataset_slice)} samples from {dataset_name} (split: {split_name}, range: {start_idx}-{end_idx-1})")
        return dataset_slice
    except Exception as e:
        logger.error(f"Error loading or slicing dataset {dataset_name}: {e}", exc_info=True)
        return None


def generate_detailed_report(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Consolidate results into a DataFrame
    pd_data = []
    for model_type, metrics_dict in all_results.items():
        row = {"Model Type": model_type}
        row.update(metrics_dict.get("metrics", {})) # Assuming 'metrics' holds CER, WER, loss etc.
        pd_data.append(row)
    results_df = pd.DataFrame(pd_data)

    report_path = os.path.join(output_dir, "detailed_comparison_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Character-Diacritic Compatibility Feature - Detailed Impact Report\n")
        f.write("=" * 70 + "\n\n")
        
        for model_type, data in all_results.items():
            metrics = data.get("metrics", {})
            gts = data.get("gts", [])
            preds = data.get("preds", [])

            f.write(f"--- Results for: {model_type} ---\n")
            for k, v in metrics.items():
                if not k.endswith("example_gts") and not k.endswith("example_preds"):
                    f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
            
            f.write("\n  Example Predictions:\n")
            for i in range(min(5, len(gts))):
                f.write(f"    GT : {gts[i]}\n")
                f.write(f"    PRD: {preds[i]}\n")
                f.write("-" * 20 + "\n")
            f.write("\n\n")

        f.write("\n--- Summary Table ---\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")

    logger.info(f"Detailed text report saved to {report_path}")

    # Plotting CER and WER
    main_metrics_to_plot = [col for col in results_df.columns if col.endswith('eval_cer') or col.endswith('eval_wer')]
    if main_metrics_to_plot:
        plot_df_main = results_df.melt(id_vars=['Model Type'], value_vars=main_metrics_to_plot, var_name='Metric', value_name='Value')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', hue='Model Type', data=plot_df_main, palette="viridis")
        plt.title('Main OCR Performance Comparison (CER/WER)')
        plt.ylabel('Error Rate')
        plt.xlabel('Metric')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ocr_performance_comparison.png"))
        plt.close()

    # Plotting Diacritic Error Counts
    diac_err_cols = [col for col in results_df.columns if col.startswith(model_type.replace(' ', '_') + '_diac_err_') or col.startswith('diac_err_')]
    # Clean up prefixes for plotting diacritic errors if they exist from different models
    cleaned_diac_err_cols_map = {}
    for col in diac_err_cols:
        for mt_prefix in [mtype.replace(' ', '_') + "_" for mtype in all_results.keys()]:
            if col.startswith(mt_prefix):
                cleaned_diac_err_cols_map[col] = col.replace(mt_prefix, "") # Base name
                break
        else: # No specific model prefix, use as is
            cleaned_diac_err_cols_map[col] = col


    if cleaned_diac_err_cols_map:
        # Create a temporary df with cleaned column names for diacritic error plotting
        temp_df_diac = results_df.copy()
        temp_df_diac.rename(columns=cleaned_diac_err_cols_map, inplace=True)
        
        # Select only the common base names of diacritic error columns
        unique_cleaned_diac_cols = sorted(list(set(cleaned_diac_err_cols_map.values())))

        plot_df_diac = temp_df_diac.melt(id_vars=['Model Type'], value_vars=unique_cleaned_diac_cols, 
                                         var_name='Diacritic Error Type', value_name='Count')
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Diacritic Error Type', y='Count', hue='Model Type', data=plot_df_diac, palette="magma")
        plt.title('Diacritic Error Analysis')
        plt.ylabel('Count')
        plt.xlabel('Error Type')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "diacritic_error_comparison.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare models with and without Character-Diacritic Compatibility.')
    parser.add_argument('--model_no_compat_path', type=str, required=True)
    parser.add_argument('--model_with_compat_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='vklinhhh/vietnamese_character_diacritic_cwl_v2')
    parser.add_argument('--dataset_split_name', type=str, default='train')
    parser.add_argument('--dev_test_start_idx', type=int, default=100000)
    parser.add_argument('--num_dev_test_samples', type=int, default=1000) # Smaller for quicker dev test
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='reports/compatibility_comparison_detailed')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_dataset_hf = get_dataset_slice(args.dataset_name, args.dev_test_start_idx, args.num_dev_test_samples, args.dataset_split_name)
    if eval_dataset_hf is None: return 1
    
    all_results_data = {} # Store metrics, gts, preds per model type

    model_paths_and_types = [
        {"path": args.model_no_compat_path, "type": "Baseline (No Compatibility)"},
        {"path": args.model_with_compat_path, "type": "With Compatibility"}
    ]

    for model_info in model_paths_and_types:
        model_path = model_info["path"]
        model_type_key = model_info["type"] # "Baseline (No Compatibility)" or "With Compatibility"
        model_type_prefix = model_type_key.replace(' ', '_').replace('(', '').replace(')', '') + "_" # For metric keys

        logger.info(f"\n--- Evaluating Model: {model_type_key} ({model_path}) ---")
        model, processor = load_model_for_eval(model_path, device)
        if model is None or processor is None:
            all_results_data[model_type_key] = {"metrics": {"status": "load_failed"}, "gts": [], "preds": []}
            continue

        # Vocabs from model config
        idx_to_char_final = {i: c for i, c in enumerate(model.config.combined_char_vocab)}
        idx_to_char_base = {i: c for i, c in enumerate(model.config.base_char_vocab)}
        idx_to_char_diac = {i: c for i, c in enumerate(model.config.diacritic_vocab)} # For diacritic name sequences
        
        blank_idx_final = model.config.blank_idx
        # Assuming blank is index 0 for base/diacritic if not specified in config explicitly
        blank_idx_base = 0 # Or get from config if stored
        blank_idx_diac = 0 # Or get from config if stored


        eval_ctc_dataset = CtcOcrDataset(eval_dataset_hf, processor, {c:i for i,c in idx_to_char_final.items()}, unk_token='[UNK]', is_training=False)
        eval_loader = DataLoader(
                    eval_ctc_dataset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=ctc_collate_fn,
                    num_workers=args.num_workers
                )
        metrics, gts, preds = evaluate_model_extended(
            model, eval_loader, device, 
            idx_to_char_final, idx_to_char_base, idx_to_char_diac,
            blank_idx_final, blank_idx_base, blank_idx_diac,
            model_type_prefix=model_type_prefix
        )
        all_results_data[model_type_key] = {"metrics": metrics, "gts": gts, "preds": preds}

        # Qualitative Analysis: Compatibility Matrix for "With Compatibility" model
        if model_type_key == "With Compatibility" and hasattr(model, 'character_diacritic_compatibility') and model.character_diacritic_compatibility:
            compat_matrix = model.character_diacritic_compatibility.compatibility_matrix.detach().cpu()
            viz_path = os.path.join(args.output_dir, "learned_compatibility_matrix.png")
            visualize_compatibility_matrix(
                compat_matrix,
                model.config.base_char_vocab,
                model.config.diacritic_vocab,
                output_path=viz_path,
                title=f"Learned Compatibility Matrix ({model_type_key})"
            )
            logger.info(f"Saved learned compatibility matrix visualization to {viz_path}")


    generate_detailed_report(all_results_data, args.output_dir)
    logger.info(f"Detailed comparison report generated in {args.output_dir}")
    return 0

if __name__ == "__main__":
    # Ensure utils can be imported if they are in a parent or utils directory
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
    sys.exit(main())