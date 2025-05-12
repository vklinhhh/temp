# utils/evaluation_reporter.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import editdistance
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import unicodedata # For character analysis later if needed
import logging
import json
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def calculate_corpus_metrics(predictions, ground_truths):
    """Calculates corpus-level CER and WER."""
    total_edit_distance_char = 0
    total_chars = 0
    total_edit_distance_word = 0
    total_words = 0

    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        logger.warning("Invalid input for metric calculation. Returning high error rates.")
        return 1.0, 1.0 # Return worst case

    for pred_str, gt_str in zip(predictions, ground_truths):
        try:
            total_edit_distance_char += editdistance.eval(pred_str, gt_str)
            total_chars += len(gt_str) # Normalize by ground truth length
        except Exception as cer_e:
            logger.warning(f"Could not calculate char edit distance for GT='{gt_str}', Pred='{pred_str}': {cer_e}")

        try:
            pred_words = pred_str.split()
            gt_words = gt_str.split()
            total_edit_distance_word += editdistance.eval(pred_words, gt_words)
            total_words += len(gt_words) # Normalize by ground truth length
        except Exception as wer_e:
            logger.warning(f"Could not calculate word edit distance for GT='{gt_str}', Pred='{pred_str}': {wer_e}")

    cer = total_edit_distance_char / total_chars if total_chars > 0 else 1.0
    wer = total_edit_distance_word / total_words if total_words > 0 else 1.0
    return cer, wer

# In utils/evaluation_reporter.py

def analyze_errors(df_results):
    """Analyzes common error types and patterns."""
    error_df = df_results[df_results['Prediction'] != df_results['GroundTruth']].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- ADDED: Calculate gt_length earlier ---
    # Calculate lengths for the *entire* results dataframe first
    if 'GroundTruth' in df_results.columns:
         df_results['gt_length'] = df_results['GroundTruth'].apply(len)
    else:
         logger.warning("GroundTruth column not found in df_results for length calculation.")
         # Handle error or return - perhaps add a dummy column?
         # For now, let's add it to error_df if it exists there.
         if 'GroundTruth' in error_df.columns:
             error_df['gt_length'] = error_df['GroundTruth'].apply(len)
         else:
             logger.error("Cannot calculate gt_length, GroundTruth column missing.")
             # Assign a dummy length if needed later, or handle the missing column downstream
             if 'gt_length' not in error_df.columns: error_df['gt_length'] = 0 
             if 'gt_length' not in df_results.columns: df_results['gt_length'] = 0 


    # Ensure the gt_length column is present in error_df if calculated on df_results
    if 'gt_length' in df_results.columns and 'gt_length' not in error_df.columns:
        error_df = df_results[df_results['Prediction'] != df_results['GroundTruth']].copy() # Re-filter to include gt_length


    analysis = {
        'total_errors': len(error_df),
        'error_rate': len(error_df) / len(df_results) if len(df_results) > 0 else 0,
        'common_char_errors': {},
        'common_word_errors': {},
        'length_mismatches': 0,
        'examples': {
            'correct': [],
            'incorrect': [],
            'longest_correct': [],
            'longest_incorrect': [],
        }
    }

    # Check if 'gt_length' is usable before proceeding with nlargest
    if 'gt_length' not in error_df.columns:
        logger.warning("gt_length column missing in error_df, cannot find longest incorrect examples.")
    if 'gt_length' not in df_results.columns:
        logger.warning("gt_length column missing in df_results, cannot find longest correct examples.")


    if analysis['total_errors'] == 0 and len(df_results) > 0:
        logger.info("No errors found in predictions.")
        # Still try to find longest correct example even if no errors
        if 'gt_length' in df_results.columns:
             longest_correct_samples = df_results[df_results['Prediction'] == df_results['GroundTruth']].nlargest(5, 'gt_length')
             analysis['examples']['longest_correct'] = longest_correct_samples.to_dict('records')
        return analysis # Return early if no errors, after finding longest correct


    char_errors = Counter()
    word_errors = Counter() # Keep this counter, even if not fully populated yet

    # Only iterate if there are errors
    if analysis['total_errors'] > 0 and 'GroundTruth' in error_df.columns and 'Prediction' in error_df.columns:
        for _, row in error_df.iterrows():
            pred = row['Prediction']
            gt = row['GroundTruth']

            # Length mismatch
            if len(pred) != len(gt):
                analysis['length_mismatches'] += 1

            # Character-level differences (simple alignment)
            for i in range(min(len(pred), len(gt))):
                if pred[i] != gt[i]:
                    char_errors[f"{gt[i]} -> {pred[i]}"] += 1

            # Word-level differences (placeholder)
            # pred_words = pred.split()
            # gt_words = gt.split()
            # if pred_words != gt_words:
            #      pass 

    analysis['common_char_errors'] = dict(char_errors.most_common(20))
    # analysis['common_word_errors'] = dict(word_errors.most_common(20)) # Implement if needed


    # --- Find examples AFTER calculating gt_length ---
    correct_samples = df_results[df_results['Prediction'] == df_results['GroundTruth']].head(5)
    incorrect_samples = error_df.head(10) # Show more incorrect examples

    # Ensure gt_length exists before calling nlargest
    if 'gt_length' in df_results.columns:
         longest_correct = df_results[df_results['Prediction'] == df_results['GroundTruth']].nlargest(5, 'gt_length')
         analysis['examples']['longest_correct'] = longest_correct.to_dict('records')
    else:
         analysis['examples']['longest_correct'] = []


    if 'gt_length' in error_df.columns:
        longest_incorrect = error_df.nlargest(5, 'gt_length')
        analysis['examples']['longest_incorrect'] = longest_incorrect.to_dict('records')
    else:
        analysis['examples']['longest_incorrect'] = []


    analysis['examples']['correct'] = correct_samples.to_dict('records')
    analysis['examples']['incorrect'] = incorrect_samples.to_dict('records')
    # The longest examples are already populated above if gt_length exists


    return analysis


def generate_visualizations(df_results, output_dir):
    """Generates plots related to performance."""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    logger.info(f"Saving visualizations to: {viz_dir}")

    # 1. Length vs CER (if CER data is available per sample - needs modification)
    # Placeholder: Distribution of prediction lengths vs GT lengths
    try:
        df_results['pred_length'] = df_results['Prediction'].apply(len)
        df_results['gt_length'] = df_results['GroundTruth'].apply(len)

        plt.figure(figsize=(12, 6))
        sns.histplot(df_results['gt_length'], color='blue', label='Ground Truth Length', kde=True, stat="density", linewidth=0, alpha=0.5)
        sns.histplot(df_results['pred_length'], color='red', label='Prediction Length', kde=True, stat="density", linewidth=0, alpha=0.5)
        plt.title('Distribution of Ground Truth vs. Prediction Lengths')
        plt.xlabel('Length (Characters)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'length_distribution.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate length distribution plot: {e}")

    # Add more plots: e.g., confidence distribution if available, confusion matrix for chars

    return viz_dir


# In utils/evaluation_reporter.py
# ... (other imports) ...

def create_html_report(report_path, summary_stats, error_analysis, viz_dir, attention_report_data=None): # Make sure this line has 5 arguments
    """Creates an HTML report summarizing the evaluation results."""
    logger.info(f"Generating HTML report at: {report_path}")

    # Basic HTML structure
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CTC OCR Evaluation Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary-box {{ border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }}
        .metrics td {{ font-weight: bold; color: #0056b3; }}
        .error {{ color: red; }}
        .correct {{ color: green; }}
        /* Adjusted image style for attention viz */
        img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ccc; }}
        .viz-container img {{ max-width: 800px; }} /* Limit standard viz size */
        .attention-container img {{ width: 300px; height: auto; }} /* Smaller size for attention grid */
        .attention-group {{ display: flex; flex-wrap: wrap; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #ccc;}}
        .attention-item {{ margin: 5px; border: 1px solid #eee; padding: 5px; text-align: center; }}
        .attention-item p {{ margin-top: 0; margin-bottom: 5px; font-weight: bold;}}
    </style>
</head>
<body>
    <h1>CTC OCR Model Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Model Path: {summary_stats.get('model_path', 'N/A')}</p>
    <p>Test Dataset: {summary_stats.get('dataset_name', 'N/A')} ({summary_stats.get('dataset_split', 'N/A')} split)</p>

    <h2>Summary Metrics</h2>
    <div class="summary-box metrics">
        <table>
            <tr><td>Total Samples</td><td>{summary_stats.get('total_samples', 'N/A')}</td></tr>
            <tr><td>Validation Loss (Avg)</td><td>{summary_stats.get('avg_loss', 'N/A'):.4f}</td></tr>
            <tr><td>Character Error Rate (CER)</td><td>{summary_stats.get('cer', 'N/A'):.4f}</td></tr>
            <tr><td>Word Error Rate (WER)</td><td>{summary_stats.get('wer', 'N/A'):.4f}</td></tr>
            <tr><td>Avg Diacritic Accuracy</td><td>{summary_stats.get('diacritic_avg_accuracy', 'N/A'):.4f}</td></tr>
            <tr><td>Dynamic Fusion Used</td><td>{summary_stats.get('use_dynamic_fusion', 'N/A')}</td></tr>
            <tr><td>Feature Enhancer Used</td><td>{summary_stats.get('use_feature_enhancer', 'N/A')}</td></tr>
        </table>
    </div>

    <h2>Error Analysis</h2>
    # ... (rest of error analysis section remains same) ...
    # Make sure error_analysis['examples'] is populated correctly
    </div>

    <h2>Visualizations</h2>
    <div class="viz-container"> 
    """
    # Add images from viz_dir (standard visualizations)
    if viz_dir and os.path.isdir(viz_dir):
        try:
            found_std_viz = False
            for filename in sorted(os.listdir(viz_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Use relative path for embedding in HTML
                    relative_path_std_viz = os.path.join(os.path.basename(viz_dir), filename) # e.g. "visualizations/file.png"
                    html += f'<h3>{os.path.splitext(filename)[0].replace("_", " ").title()}</h3>\n'
                    html += f'<img src="{relative_path_std_viz}" alt="{filename}">\n'
                    found_std_viz = True
            if not found_std_viz:
                 html += "<p>No standard visualization images (.png, .jpg) found in the visualization directory.</p>"
        except Exception as e:
            html += f"<p>Error loading standard visualizations: {e}</p>"
    else:
        html += "<p>Standard visualization directory not found or specified.</p>"
    html += "</div>" # End of viz-container

    # --- Add Attention Visualizations Section ---
    if attention_report_data:
        html += """
        <h2>Diacritic Attention Visualizations</h2>
        <p>Visualizations showing where the 'VisualDiacriticAttention' module focuses (Above, Middle, Below regions). Limited to a few samples.</p>
        """
        for item in attention_report_data:
            html += f"<h3>Sample {item['sample_idx']} (GT: '{item['gt']}', Pred: '{item['pred']}')</h3>\n"
            # Use attention-group for flex layout
            html += "<div class='attention-group'>\n" 
            if item.get('image_paths'):
                for img_path in item['image_paths']: # image_paths are relative to output_dir
                    # Extract region name from filename, handle potential errors
                    try:
                        region_name = os.path.basename(img_path).split('_region')[-1].split('.')[0]
                        if region_name.isdigit(): # Check if it's just the index
                             region_name = ["Above", "Middle", "Below"][int(region_name)]
                    except (IndexError, ValueError):
                        region_name = "Unknown Region"

                    html += f"""
                    <div class='attention-item'>
                        <p>{region_name}</p>
                        <img src='{img_path}' alt='{os.path.basename(img_path)}'>
                    </div>
                    """
            else:
                 html += "<p>Attention images not available for this sample.</p>"
            html += "</div>\n" # End attention-group
    # --- END: Attention Visualizations Section ---


    html += """
    <h2>Examples</h2>
    """
    # Add examples tables (ensure error_analysis['examples'] is populated)
    for category, examples in error_analysis.get('examples', {}).items():
        if examples:
            html += f"<h3>{category.replace('_', ' ').title()}</h3>\n<table>\n"
            html += "<tr><th>#</th><th>Ground Truth</th><th>Prediction</th><th>Correct?</th><th>Length (GT)</th></tr>\n"
            for i, ex in enumerate(examples):
                # Check if both GT and Pred exist before comparing
                gt_text = ex.get('GroundTruth', '')
                pred_text = ex.get('Prediction', '')
                is_correct = (gt_text == pred_text) and gt_text is not None # Handle potential None values
                correct_class = "correct" if is_correct else "error"
                gt_len = ex.get('gt_length', 'N/A') # Use the precalculated length
                html += f"<tr><td>{i+1}</td><td>{gt_text}</td><td>{pred_text}</td><td class='{correct_class}'>{is_correct}</td><td>{gt_len}</td></tr>\n"
            html += "</table>\n"
        else:
            html += f"<h3>{category.replace('_', ' ').title()}</h3>\n<p>No examples in this category.</p>\n"


    html += """
</body>
</html>
    """

    # Write report
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info("HTML report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write HTML report: {e}")