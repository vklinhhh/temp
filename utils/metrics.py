# utils/metrics.py
import jiwer
import logging
import wandb

logger = logging.getLogger(__name__)

def calculate_cer_wer(predictions, references):
    """
    Calculates Character Error Rate (CER) and Word Error Rate (WER).

    Args:
        predictions (list of str): List of predicted transcriptions.
        references (list of str): List of ground truth transcriptions.

    Returns:
        dict: A dictionary containing 'cer' and 'wer'.
    """
    if not predictions or not references or len(predictions) != len(references):
        logger.warning("Invalid input for CER/WER calculation. Predictions or references empty or mismatched lengths.")
        return {'cer': 1.0, 'wer': 1.0} # Return worst case if input is bad

    try:
        # Default transformation for jiwer (you might want to customize this for specific languages)
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.SentencesToListOfWords(),
            jiwer.RemoveEmptyStrings()
        ])

        wer = jiwer.wer(references, predictions, truth_transform=transformation, hypothesis_transform=transformation)
        mer = jiwer.mer(references, predictions, truth_transform=transformation, hypothesis_transform=transformation) # Match Error Rate, often similar to WER
        wil = jiwer.wil(references, predictions, truth_transform=transformation, hypothesis_transform=transformation) # Word Information Lost

        # For CER, process character by character
        # Create a basic character transformation
        char_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            # jiwer.RemovePunctuation(), # Optional: Keep or remove punctuation for CER based on your needs
            jiwer.SentencesToListOfCharacters() # Key change for CER
        ])
        cer = jiwer.cer(references, predictions, truth_transform=char_transformation, hypothesis_transform=char_transformation)

    except Exception as e:
        logger.error(f"Error calculating CER/WER with jiwer: {e}")
        # Fallback to simple calculation if jiwer fails, or return error values
        # This basic fallback might not be as accurate or robust as jiwer.
        # For simplicity, returning error values.
        return {'cer': 1.0, 'wer': 1.0, 'mer': 1.0, 'wil': 1.0}


    return {
        'cer': cer if cer is not None else 1.0,
        'wer': wer if wer is not None else 1.0,
        'mer': mer if mer is not None else 1.0, # Match Error Rate
        'wil': wil if wil is not None else 1.0  # Word Information Lost
    }

def log_metrics_to_wandb(wandb_run, metrics_dict, epoch, global_step, commit=True, prefix="val"):
    """
    Logs a dictionary of metrics to Weights & Biases.

    Args:
        wandb_run: Active wandb run object.
        metrics_dict (dict): Dictionary of metrics to log.
        epoch (int): Current epoch number.
        global_step (int): Current global training step.
        commit (bool): Whether to commit the log immediately.
        prefix (str): Prefix for the metric names (e.g., "val", "test").
    """
    if wandb_run is None:
        return

    log_payload = {}
    for key, value in metrics_dict.items():
        # Ensure correct prefixing if not already done
        if not key.startswith(f"{prefix}/"):
            log_payload[f"{prefix}/{key}"] = value
        else:
            log_payload[key] = value # Already prefixed (e.g. from compute_ctc_validation_metrics)
    
    # Add epoch and global_step for context if not already part of the metrics keys
    if f"{prefix}/epoch" not in log_payload: # Avoid overwriting if already there
        log_payload[f"{prefix}/epoch_context"] = epoch 
    if f"{prefix}/global_step" not in log_payload:
        log_payload[f"{prefix}/global_step_context"] = global_step

    try:
        wandb_run.log(log_payload, step=global_step, commit=commit)
    except Exception as e:
        logger.error(f"Failed to log metrics to WandB: {e}")