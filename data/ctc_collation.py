# data/ctc_collation.py
import torch
import logging

logger = logging.getLogger(__name__)
def ctc_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    try:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
    except Exception as e:
        logger.error(f"Error stacking pixel_values: {e}.")
        shapes = [item['pixel_values'].shape for item in batch]
        logger.error(f"Pixel value shapes in failing batch: {shapes}")
        # It's better to raise an error or return None to be handled by DataLoader's error handling
        # Forcing a skip here might hide issues. If a batch *must* be returned, it needs to be valid.
        return None 

    labels = [item['labels'] for item in batch]
    label_lengths = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0
    )
    texts = [item.get('text', '') for item in batch]

    collated_batch = {
        'pixel_values': pixel_values,
        'labels': padded_labels,
        'label_lengths': label_lengths,
        'texts': texts
    }
    
    # Collect original images if present (typically for non-training mode)
    # Ensure all items in 'batch' are checked for 'original_image_pil'
    # and that the list corresponds to the filtered batch.
    if 'original_image_pil' in batch[0]: # Check if the key exists in the first valid item
        original_images_pil = [item.get('original_image_pil') for item in batch]
        # Filter out None if some items didn't have it (though logic above tries to ensure they do for eval)
        # For simplicity, we assume if one has it, all (valid) items in the batch should have it for eval.
        if all(img is not None for img in original_images_pil):
             collated_batch['original_images_pil'] = original_images_pil
        else:
            logger.warning("Not all items in the batch had 'original_image_pil'. It will not be added to the collated batch.")


    return collated_batch