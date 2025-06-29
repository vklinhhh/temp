# Vietnamese OCR with Hierarchical CTC and Dynamic Feature Enhancement

A specialized OCR model designed for Vietnamese text recognition, with advanced components for handling the complex diacritical mark system used in Vietnamese writing.

## Key Features

- **Hierarchical Multi-Stage Architecture**: Specialized components for base characters and diacritical marks
- **Dynamic Multi-Scale Fusion**: Adaptively combines features from different vision encoder layers
- **Local Feature Enhancer**: Targeted enhancement for diacritical mark detection
- **Three-Head Classification System**: Separate classification paths for base characters, diacritics, and combined characters

## Architecture Overview

The architecture consists of several specialized components designed to handle the complexity of Vietnamese text recognition:

### Core Components

1. **Vision Encoder**: Extracts multi-scale visual features from input images using a pre-trained vision encoder (based on TrOCR/ViT)

2. **Dynamic Multi-Scale Fusion**:
   - Adaptively combines features from different encoder layers
   - Uses global content analysis to determine importance weights for each layer
   - Employs position-specific gating for fine-grained feature selection
   - Balances global character recognition with local detail preservation

3. **Local Feature Enhancer**:
   - Specialized in enhancing diacritical mark features
   - Uses region-specific attention (above/middle/below character regions)
   - Contains diacritic prototypes for improved recognition
   - Makes subtle diacritical marks more prominent in the feature representation

4. **Transformer Encoder**:
   - Processes enhanced features through multiple transformer layers
   - Captures sequential dependencies in text

5. **Hierarchical Classification Heads**:
   - **Base Character Classifier**: Recognizes the fundamental character (e.g., 'a', 'e', 'o')
   - **Diacritic Classifier**: Identifies the diacritical mark (e.g., 'acute', 'grave', 'hook')
   - **Combined Character Classifier**: Predicts the full combined character (e.g., 'á', 'à', 'ả')

6. **CTC Loss**: Enables alignment-free sequence prediction using Connectionist Temporal Classification

### Key Innovations

- **Diacritic-Aware Feature Processing**: Components specifically designed for Vietnamese diacritics
- **Adaptive Multi-Scale Feature Integration**: Dynamically adjusts feature importance based on content
- **Hierarchical Classification Approach**: Decomposes the recognition problem into simpler sub-tasks

## Model Performance

The model demonstrates strong performance on Vietnamese OCR tasks, with particular strengths in:

- Accurate recognition of complex diacritical marks
- Handling of visually similar characters with different diacritics
- Robust performance across various fonts and writing styles

Character Error Rate (CER) and Word Error Rate (WER) metrics show significant improvements over baseline models, especially for text with multiple diacritical marks.

## Visualization Tools

The repository includes tools for visualizing model attention and feature importance:

- **Grad-CAM Visualization**: Shows which parts of the image the model focuses on for specific predictions
- **Attention Visualization**: Displays the region-specific attention maps for diacritical mark detection
- **Evaluation Reports**: Generates comprehensive HTML reports of model performance with visualizations

## Usage Examples

### Training

```python
python scripts/train_hierarchical_ctc.py \
    --dataset_name vklinhhh/vietnamese_character_diacritic_cwl_v2 \
    --vision_encoder microsoft/trocr-base-handwritten \
    --output_dir outputs/dynamic_fusion_large \
    --fusion_layers "-1,-4" \
    --use_dynamic_fusion \
    --use_feature_enhancer \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 30 \
    --wandb_project vietnam-ocr
```

```
python -m scripts.train_hierarchical_ctc   --dataset_name vklinhhh/vnhwt_opt_3    --output_dir outputs/opt_3_sequential_ablation    --use_dynamic_fusion --use_feature_enhancer   --transformer_d_model 768   --transformer_nhead 12   --num_transformer_layers 6   --shared_hidden_size 768   --batch_size 16   --learning_rate 5e-5   --discriminative_lr   --encoder_lr_factor 0.1   --warmup_ratio 0.05   --weight_decay 0.02   --epochs 50   --use_character_diacritic_compatibility   --use_amp   --fusion_method concat_proj --log_compatibility_interval 15000 --wandb_proj opt_3_sequential --early_stopping_patience 7 --hierarchical_mode sequential  --no_middle_diacritic_conditioning
```


### Evaluation

```python
python scripts/evaluate_hierarchical_ctc.py \
    --model_path outputs/dynamic_fusion_large/best_model_hf \
    --combined_char_vocab_path outputs/dynamic_fusion_large/combined_char_vocab.json \
    --dataset_name vklinhhh/test_vietnamese_cwl \
    --output_dir evaluation_results
```

### Visualization

```python
python scripts/visualize_all_layers_grad_cam.py \
    --model_path outputs/dynamic_fusion_large/best_model_hf \
    --combined_char_vocab_path outputs/dynamic_fusion_large/combined_char_vocab.json \
    --image_path path/to/image.jpg \
    --output_path_prefix visualizations/output \
    --grad_cam_target_diacritic acute
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- albumentations
- editdistance
- cv2
- wandb (optional, for logging)

## Project Structure

```
├── data/              # Dataset handling
│   ├── ctc_collation.py
│   ├── ctc_ocr_dataset.py
│   └── __init__.py
├── model/             # Model architecture
│   ├── diacritic_attention.py
│   ├── dynamic_fusion.py
│   ├── hierarchical_ctc_model.py
│   └── __init__.py
├── training/          # Training and validation
│   ├── ctc_trainer.py
│   ├── ctc_validation.py
│   └── __init__.py
├── utils/             # Utility functions
│   ├── ctc_utils.py
│   ├── evaluation_reporter.py
│   ├── optimizers.py
│   ├── schedulers.py
│   └── __init__.py
├── scripts/           # Training and evaluation scripts
│   ├── train_hierarchical_ctc.py
│   ├── evaluate_hierarchical_ctc.py
│   ├── visualize_single_image.py
│   └── visualize_all_layers_grad_cam.py
```

## Citation

If you use this work in your research, please cite:

```
@article{vietnamese-hierarchical-ctc,
  title={Vietnamese OCR with Hierarchical CTC and Dynamic Feature Enhancement},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

- This project builds upon the TrOCR architecture from Microsoft
- Special thanks to the contributors of the Vietnamese Character Dataset
