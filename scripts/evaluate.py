#!/usr/bin/env python
# evaluate_ocr.py
"""
Evaluation script for a Vietnamese OCR model using CTC decoding.
Loads a test dataset from Hugging Face and evaluates model performance.
"""

import os
import sys
import argparse
import torch
import logging
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import necessary components based on your project structure
from utils.ctc_utils import CTCDecoder
from data.ctc_ocr_dataset import CtcOcrDataset
from data.ctc_collation import ctc_collate_fn
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ocr_evaluation.log'),
    ],
)
logger = logging.getLogger('OCR-Evaluation')

def evaluate_model(
    model_path,
    dataset_name,
    dataset_split="train",
    output_dir="evaluation_results",
    batch_size=16,
    num_workers=4,
    device=None
):
    """
    Evaluate OCR model on a test dataset.
    
    Args:
        model_path: Path to the trained model
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        device: Device to run evaluation on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    try:
        logger.info(f"Loading model from {model_path}")
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Get processor from model
        processor = model.processor
        
        # Create CTC decoder
        if hasattr(model, 'combined_char_vocab'):
            idx_to_char = {i: c for i, c in enumerate(model.combined_char_vocab)}
            blank_idx = 0  # Assuming blank is at index 0
            decoder = CTCDecoder(idx_to_char_map=idx_to_char, blank_idx=blank_idx)
            logger.info(f"Created decoder with vocabulary size: {len(idx_to_char)}")
        else:
            logger.error("Model does not have combined_char_vocab attribute!")
            return
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return
    
    # Load dataset
    try:
        logger.info(f"Loading dataset: {dataset_name}, split: {dataset_split}")
        hf_dataset = load_dataset(dataset_name)
        if dataset_split not in hf_dataset:
            available_splits = list(hf_dataset.keys())
            logger.warning(f"Split '{dataset_split}' not found. Available splits: {available_splits}")
            if len(available_splits) > 0:
                dataset_split = available_splits[0]
                logger.warning(f"Using '{dataset_split}' split instead.")
            else:
                raise ValueError("No valid splits found in dataset")
        
        test_ds = hf_dataset[dataset_split]
        logger.info(f"Dataset loaded with {len(test_ds)} samples")
        
        # Check that dataset has required columns
        if 'image' not in test_ds.column_names:
            raise ValueError("Dataset must have 'image' column")
        if 'label' not in test_ds.column_names:
            logger.warning("'label' column not found, checking alternatives...")
            label_col = next((col for col in ['text', 'word', 'annotation'] if col in test_ds.column_names), None)
            if label_col:
                logger.info(f"Using '{label_col}' column as label")
                # Create a new dataset with 'label' column
                test_ds = test_ds.map(lambda example: {'label': example[label_col]})
            else:
                raise ValueError("Dataset must have 'label' or equivalent column")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return
    
    # Create dataset wrapper and dataloader
    try:
        char_to_idx = {c: i for i, c in enumerate(model.combined_char_vocab)}
        test_dataset = CtcOcrDataset(
            test_ds, 
            processor, 
            char_to_idx, 
            unk_token='[UNK]', 
            is_training=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ctc_collate_fn,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda')
        )
        logger.info(f"Created DataLoader with {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"Error creating dataset/dataloader: {e}", exc_info=True)
        return
    
    # Run evaluation
    all_predictions = []
    all_ground_truths = []
    character_errors = 0
    total_characters = 0
    word_errors = 0
    total_words = 0
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch is None:
                continue
                
            try:
                # Move data to device
                pixel_values = batch['pixel_values'].to(device)
                texts_gt = batch['texts']  # Original text strings
                
                # Forward pass
                outputs = model(pixel_values=pixel_values)
                logits = outputs['logits']
                
                # Decode predictions
                predicted_texts = decoder(logits)
                
                # Store results
                all_predictions.extend(predicted_texts)
                all_ground_truths.extend(texts_gt)
                
                # Calculate character and word error rates on-the-fly
                for pred, gt in zip(predicted_texts, texts_gt):
                    # Character error
                    import editdistance
                    char_dist = editdistance.eval(pred, gt)
                    character_errors += char_dist
                    total_characters += len(gt)
                    
                    # Word error
                    pred_words = pred.split()
                    gt_words = gt.split()
                    word_dist = editdistance.eval(pred_words, gt_words)
                    word_errors += word_dist
                    total_words += len(gt_words)
                
                # Log a few examples from the first batch
                if batch_idx == 0:
                    for i in range(min(3, len(predicted_texts))):
                        logger.info(f"Example {i}:")
                        logger.info(f"  Ground truth: '{texts_gt[i]}'")
                        logger.info(f"  Prediction  : '{predicted_texts[i]}'")
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Calculate final metrics
    if total_characters > 0:
        cer = character_errors / total_characters
    else:
        cer = 1.0
        
    if total_words > 0:
        wer = word_errors / total_words
    else:
        wer = 1.0
    
    logger.info(f"Evaluation complete!")
    logger.info(f"Character Error Rate (CER): {cer:.4f}")
    logger.info(f"Word Error Rate (WER): {wer:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'GroundTruth': all_ground_truths,
        'Prediction': all_predictions
    })
    
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved detailed results to: {results_path}")
    
    # Save summary metrics
    metrics = {
        'cer': cer,
        'wer': wer,
        'total_samples': len(all_ground_truths),
        'total_characters': total_characters,
        'total_words': total_words
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics summary to: {metrics_path}")
    
    # Create a simple HTML report
    create_html_report(
        output_dir=output_dir,
        metrics=metrics,
        examples=list(zip(all_ground_truths[:50], all_predictions[:50]))  # First 50 examples
    )
    
    return metrics

def create_html_report(output_dir, metrics, examples):
    """Create a simple HTML report with evaluation results."""
    report_path = os.path.join(output_dir, "evaluation_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metrics {{ font-size: 18px; margin: 20px 0; }}
            .metric-value {{ font-weight: bold; color: #0056b3; }}
            .error {{ color: red; }}
            .correct {{ color: green; }}
        </style>
    </head>
    <body>
        <h1>OCR Evaluation Report</h1>
        
        <div class="metrics">
            <p>Character Error Rate (CER): <span class="metric-value">{metrics['cer']:.4f}</span></p>
            <p>Word Error Rate (WER): <span class="metric-value">{metrics['wer']:.4f}</span></p>
            <p>Total Samples: <span class="metric-value">{metrics['total_samples']}</span></p>
            <p>Total Characters: <span class="metric-value">{metrics['total_characters']}</span></p>
            <p>Total Words: <span class="metric-value">{metrics['total_words']}</span></p>
        </div>
        
        <h2>Example Predictions</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Ground Truth</th>
                <th>Prediction</th>
                <th>Correct?</th>
            </tr>
    """
    
    import editdistance
    
    for i, (gt, pred) in enumerate(examples):
        is_correct = gt == pred
        row_class = "correct" if is_correct else "error"
        
        html_content += f"""
            <tr class="{row_class}">
                <td>{i+1}</td>
                <td>{gt}</td>
                <td>{pred}</td>
                <td>{"✓" if is_correct else "✗"}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Created HTML report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--dataset_name", type=str, default="vklinhhh/test",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()