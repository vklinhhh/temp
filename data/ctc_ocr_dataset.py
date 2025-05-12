import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import logging
import numpy as np
# import unicodedata # Not used in this file's current state
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

logger = logging.getLogger(__name__)

class CtcOcrDataset(Dataset):
    def __init__(self, hf_dataset, processor, char_to_idx_map, unk_token='[UNK]', ignore_case=False, is_training=False):
        self.dataset = hf_dataset
        self.processor = processor
        self.char_to_idx = char_to_idx_map
        self.ignore_case = ignore_case
        self.unk_token = unk_token
        self.is_training = is_training

        if self.unk_token not in self.char_to_idx:
             raise ValueError(f"Unknown token '{self.unk_token}' not found in char_to_idx_map.")
        self.unk_idx = self.char_to_idx[self.unk_token]
        self.blank_idx = 0

        if self.is_training:
            self.augment_transform = A.Compose([
                A.Rotate(limit=7, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), shear=(-5, 5), p=0.7, mode=cv2.BORDER_REPLICATE),
                A.Perspective(scale=(0.01, 0.05), p=0.3, pad_mode=cv2.BORDER_REPLICATE),
                A.ElasticTransform(alpha=1, sigma=15, alpha_affine=15, p=0.3, border_mode=cv2.BORDER_REPLICATE),
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3, border_mode=cv2.BORDER_REPLICATE),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 5), p=1.0)
                ], p=0.4),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80,120),p=0.3),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                    A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=1.0),
                ], p=0.3),
            ])
            logger.info("Training augmentations enabled.")
        else:
            self.augment_transform = None
            logger.info("Training augmentations disabled (validation/test mode).")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]
            original_image_pil = None

            image_input_for_processor = example['image']
            try:
                if isinstance(image_input_for_processor, str):
                    original_image_pil = Image.open(image_input_for_processor).convert("RGB")
                elif isinstance(image_input_for_processor, np.ndarray):
                    original_image_pil = Image.fromarray(image_input_for_processor).convert("RGB")
                elif isinstance(image_input_for_processor, Image.Image):
                    original_image_pil = image_input_for_processor.convert('RGB') if image_input_for_processor.mode != 'RGB' else image_input_for_processor.copy()
                else:
                    raise TypeError(f"Unsupported image type: {type(image_input_for_processor)}")
                
                image_to_process = original_image_pil.copy() # Process a copy

            except (FileNotFoundError, UnidentifiedImageError, TypeError, ValueError) as img_err:
                logger.warning(f"Skipping sample {idx} due to image loading/conversion error: {img_err}")
                return None
            except Exception as e:
                 logger.error(f"Unexpected error loading image for sample {idx}: {e}", exc_info=True)
                 return None

            if self.is_training and self.augment_transform:
                image_np = np.array(image_to_process)
                try:
                    augmented = self.augment_transform(image=image_np)
                    image_aug_np = augmented['image']
                    image_to_process = Image.fromarray(image_aug_np)
                except Exception as aug_err:
                    logger.warning(f"Augmentation failed for sample {idx}: {aug_err}. Using un-augmented image.")
                                
            try:
                pixel_values = self.processor(image_to_process, return_tensors="pt").pixel_values.squeeze(0)
            except Exception as proc_err:
                logger.warning(f"Hugging Face processor failed for sample {idx} (text: '{example.get('label', example.get('word', example.get('text', 'N/A')))}'): {proc_err}. Skipping.")
                return None

            text_label = example.get('label', example.get('word', example.get('text')))
            if text_label is None or not isinstance(text_label, str):
                logger.warning(f"Skipping sample {idx} due to missing or invalid text label.")
                return None

            if self.ignore_case: text_label = text_label.lower()
            label_indices = [self.char_to_idx.get(char, self.unk_idx) for char in text_label]
            if not label_indices:
                 logger.warning(f"Sample {idx} resulted in empty label indices for text: '{text_label}'. Skipping.")
                 return None

            item_dict = {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label_indices, dtype=torch.long),
                "text": text_label
            }
            # Add original_image_pil for evaluation/visualization purposes
            if not self.is_training and original_image_pil is not None:
                item_dict["original_image_pil"] = original_image_pil
            
            return item_dict

        except Exception as e:
            logger.error(f"Unexpected error getting item {idx}: {e}", exc_info=True)
            return None

