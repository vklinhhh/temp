# model/hierarchical_ctc_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel,
    PreTrainedModel,
    PretrainedConfig
)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os
import json
import logging
import math
# Assuming these are in the same directory or accessible via PYTHONPATH
from .diacritic_attention import (
    VisualDiacriticAttention,
    CharacterDiacriticCompatibility,
    FewShotDiacriticAdapter
)
from .dynamic_fusion import (
    DynamicMultiScaleFusion,
    LocalFeatureEnhancer
)

logger = logging.getLogger(__name__)

# --- Configuration Class (HierarchicalCtcOcrConfig) ---
class HierarchicalCtcOcrConfig(PretrainedConfig):
    model_type = "hierarchical_ctc_transformer"

    def __init__(
        self,
        vision_encoder_name='microsoft/trocr-base-handwritten',
        base_char_vocab=None, diacritic_vocab=None, combined_char_vocab=None,
        vision_encoder_layer_indices=[-1, -4], feature_fusion_method="concat_proj",
        use_dynamic_fusion=False, use_feature_enhancer=False,
        num_transformer_encoder_layers=4, transformer_d_model=512,
        transformer_nhead=8, transformer_dim_feedforward=2048,
        transformer_dropout=0.1, positional_encoding_type="sinusoidal_1d",
        max_seq_len_for_pos_enc=768, shared_hidden_size=512,
        num_shared_layers=1, conditioning_method="concat_proj",
        classifier_dropout=0.1, blank_idx=0, vision_encoder_config=None,
        use_visual_diacritic_attention=False, use_character_diacritic_compatibility=False,
        use_few_shot_diacritic_adapter=False, num_few_shot_prototypes=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder_name = vision_encoder_name
        self.base_char_vocab = base_char_vocab if base_char_vocab else []
        self.diacritic_vocab = diacritic_vocab if diacritic_vocab else []
        self.combined_char_vocab = combined_char_vocab if combined_char_vocab else []
        self.base_char_vocab_size = len(self.base_char_vocab)
        self.diacritic_vocab_size = len(self.diacritic_vocab)
        self.combined_char_vocab_size = len(self.combined_char_vocab)
        self.vision_encoder_layer_indices = sorted(list(set(vision_encoder_layer_indices)))
        self.feature_fusion_method = feature_fusion_method
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_feature_enhancer = use_feature_enhancer
        self.num_transformer_encoder_layers = num_transformer_encoder_layers
        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.transformer_dropout = transformer_dropout
        self.positional_encoding_type = positional_encoding_type
        self.max_seq_len_for_pos_enc = max_seq_len_for_pos_enc
        self.shared_hidden_size = shared_hidden_size
        self.num_shared_layers = num_shared_layers
        self.conditioning_method = conditioning_method
        self.classifier_dropout = classifier_dropout
        self.blank_idx = blank_idx
        self.vision_encoder_config = vision_encoder_config # Can be dict or object
        self.use_visual_diacritic_attention = use_visual_diacritic_attention
        self.use_character_diacritic_compatibility = use_character_diacritic_compatibility
        self.use_few_shot_diacritic_adapter = use_few_shot_diacritic_adapter
        self.num_few_shot_prototypes = num_few_shot_prototypes
        if not self.combined_char_vocab:
            logger.warning("Combined character vocabulary is empty during config init.")

# --- Positional Encoding Classes ---
class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)

# --- Model Class (HierarchicalCtcTransformerOcrModel - Base for MultiScale) ---
class HierarchicalCtcTransformerOcrModel(PreTrainedModel):
    config_class = HierarchicalCtcOcrConfig

    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        self.config = config
        if not config.combined_char_vocab: raise ValueError("Combined vocab missing for model init.")
        if config.conditioning_method != 'none' and (not config.base_char_vocab or not config.diacritic_vocab):
            logger.warning("Base/Diac vocabs might be missing for conditioning.")

        self.base_char_vocab = config.base_char_vocab
        self.diacritic_vocab = config.diacritic_vocab
        self.combined_char_vocab = config.combined_char_vocab

        logger.info(f"Initializing {self.__class__.__name__} with config: {config.model_type}")

        # --- Load Base Components ---
        try:
            self.processor = AutoProcessor.from_pretrained(config.vision_encoder_name, trust_remote_code=True)
            base_v_model = VisionEncoderDecoderModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)
            self.vision_encoder = base_v_model.encoder
            # Ensure vision_encoder_config is an object
            if isinstance(config.vision_encoder_config, dict):
                vision_config_class = self.vision_encoder.config.__class__ # Get specific config class
                self.config.vision_encoder_config = vision_config_class(**config.vision_encoder_config)
            elif self.config.vision_encoder_config is None:
                 self.config.vision_encoder_config = self.vision_encoder.config

            logger.info("Processor and Vision Encoder loaded.")
            del base_v_model
        except Exception as e:
            logger.error(f"Failed loading base model components: {e}", exc_info=True); raise

        # --- Feature Fusion Layers (Dynamic or Static) ---
        encoder_hidden_size = self.config.vision_encoder_config.hidden_size
        num_fusion_layers = len(config.vision_encoder_layer_indices)
        self.dynamic_fusion = None; self.fusion_projection = None; self.pre_fusion_projections = None; self.fusion_bilinear = None
        sequence_model_input_size = config.transformer_d_model # Target dimension for fusion output

        if config.use_dynamic_fusion and num_fusion_layers > 1:
            logger.info(f"Using Dynamic Multi-Scale Fusion ({num_fusion_layers} layers -> {config.transformer_d_model} dim)")
            self.dynamic_fusion = DynamicMultiScaleFusion(encoder_hidden_size, num_fusion_layers, config.transformer_d_model)
            self.post_static_fusion_norm = None # Not used with dynamic fusion
        elif config.feature_fusion_method == "concat_proj" and num_fusion_layers > 1:
            self.fusion_projection = nn.Linear(encoder_hidden_size * num_fusion_layers, config.transformer_d_model)
            logger.info(f"Using 'concat_proj' fusion (In: {encoder_hidden_size * num_fusion_layers}, Out: {config.transformer_d_model})")
        elif config.feature_fusion_method == "add" and num_fusion_layers > 1:
            if encoder_hidden_size != config.transformer_d_model:
                self.pre_fusion_projections = nn.ModuleList([nn.Linear(encoder_hidden_size, config.transformer_d_model) for _ in range(num_fusion_layers)])
            logger.info(f"Using 'add' feature fusion (Out: {config.transformer_d_model})")
        elif config.feature_fusion_method == "bilinear" and num_fusion_layers == 2:
             self.fusion_bilinear = nn.Bilinear(encoder_hidden_size, encoder_hidden_size, config.transformer_d_model)
             logger.info(f"Using 'bilinear' feature fusion (Out: {config.transformer_d_model})")
        else: # 'none' or single layer selected, or fallback for static fusion
             if encoder_hidden_size != config.transformer_d_model:
                  self.fusion_projection = nn.Linear(encoder_hidden_size, config.transformer_d_model)
                  logger.info(f"Projecting single/last selected encoder layer ({encoder_hidden_size}) to {config.transformer_d_model}")
             # If encoder_hidden_size == config.transformer_d_model, sequence_model_input_size is already encoder_hidden_size.
             # No explicit fusion_projection needed if dimensions match and it's a single layer direct use.
             # sequence_model_input_size is correctly set to config.transformer_d_model at the start.

        # Added: LayerNorm after static fusion, if static fusion is configured.
        # This norm operates on features of size config.transformer_d_model.
        if not config.use_dynamic_fusion:
            self.post_static_fusion_norm = nn.LayerNorm(config.transformer_d_model)
            logger.info(f"Added LayerNorm after static fusion output (dim: {config.transformer_d_model})")
        # else: self.post_static_fusion_norm = None # Already handled above

        # --- Local Feature Enhancer ---
        self.feature_enhancer = None
        if config.use_feature_enhancer:
            logger.info("Using Local Feature Enhancer")
            # Input to feature enhancer is the output of fusion, which is config.transformer_d_model
            self.feature_enhancer = LocalFeatureEnhancer(config.transformer_d_model, config.diacritic_vocab_size)


        # --- Positional Encoding ---
        # Input to pos_encoder is config.transformer_d_model
        if config.positional_encoding_type == "sinusoidal_1d": self.pos_encoder = SinusoidalPositionalEncoding1D(config.transformer_d_model, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        elif config.positional_encoding_type == "learned_1d": self.pos_encoder = LearnedPositionalEncoding1D(config.transformer_d_model, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        else: self.pos_encoder = nn.Identity()
        logger.info(f"Positional Encoding: {config.positional_encoding_type}")

        # --- Transformer Encoder ---
        # Input to transformer_encoder is config.transformer_d_model
        logger.info(f"Adding {config.num_transformer_encoder_layers} Transformer Encoder layers (Dim: {config.transformer_d_model})")
        if config.transformer_d_model % config.transformer_nhead != 0: raise ValueError("Transformer d_model must be divisible by nhead")
        tf_encoder_layer = TransformerEncoderLayer(config.transformer_d_model, config.transformer_nhead, config.transformer_dim_feedforward, config.transformer_dropout, F.gelu, batch_first=True)
        self.transformer_encoder = TransformerEncoder(tf_encoder_layer, config.num_transformer_encoder_layers, nn.LayerNorm(config.transformer_d_model))
        transformer_output_size = config.transformer_d_model


        # --- Shared Feature Layer ---
        shared_layers_modules = []
        current_shared_size = transformer_output_size
        for i in range(config.num_shared_layers):
            out_s = config.shared_hidden_size
            shared_layers_modules.extend([nn.Linear(current_shared_size, out_s), nn.LayerNorm(out_s), nn.GELU(), nn.Dropout(config.classifier_dropout)])
            current_shared_size = out_s
        self.shared_layer = nn.Sequential(*shared_layers_modules) if config.num_shared_layers > 0 else nn.Identity()
        shared_output_size = current_shared_size
        logger.info(f"Shared Layer(s): {config.num_shared_layers}, Output Dim: {shared_output_size}")

        # --- Hierarchical Heads ---
        self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
        logger.info(f"Base Classifier (In: {shared_output_size}, Out: {config.base_char_vocab_size})")

        self.diacritic_gate = None; self.diacritic_condition_proj = None
        diacritic_head_input_size = shared_output_size
        if config.conditioning_method == 'concat_proj':
            self.diacritic_condition_proj = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.LayerNorm(shared_output_size), nn.GELU(), nn.Dropout(config.classifier_dropout))
            logger.info(f"'concat_proj' conditioning for diacritic head -> {shared_output_size} dim.")
        elif config.conditioning_method == 'gate':
            self.diacritic_gate = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.Sigmoid())
            logger.info(f"'gate' conditioning for diacritic head -> {shared_output_size} dim.")
        else: logger.info(f"'none' conditioning for diacritic head -> {shared_output_size} dim.")

        self.visual_diacritic_attention = None; self.character_diacritic_compatibility = None; self.few_shot_diacritic_adapter = None
        if config.use_visual_diacritic_attention: self.visual_diacritic_attention = VisualDiacriticAttention(diacritic_head_input_size, config.diacritic_vocab_size); logger.info("Visual Diacritic Attn: ON")
        if config.use_character_diacritic_compatibility: self.character_diacritic_compatibility = CharacterDiacriticCompatibility(config.base_char_vocab_size, config.diacritic_vocab_size, diacritic_head_input_size, self.base_char_vocab, self.diacritic_vocab); logger.info("Char-Diac Compat: ON")
        if config.use_few_shot_diacritic_adapter: self.few_shot_diacritic_adapter = FewShotDiacriticAdapter(diacritic_head_input_size, config.diacritic_vocab_size, config.num_few_shot_prototypes); logger.info("Few-Shot Adapter: ON")

        self.diacritic_classifier = nn.Linear(diacritic_head_input_size, config.diacritic_vocab_size)
        logger.info(f"Diacritic Classifier (In: {diacritic_head_input_size}, Out: {config.diacritic_vocab_size})")

        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)
        logger.info(f"Final Combined Classifier (In: {shared_output_size}, Out: {config.combined_char_vocab_size})")

        # --- Grad-CAM Hooks ---
        self.activation_hook_handles = []
        self.activations = None
        # self.gradients will be read from self.activations.grad

        self._init_weights()

    def _init_weights(self): # Simplified for brevity, ensure it covers all custom layers
        logger.debug("Initializing weights for custom layers...")
        # Standard PyTorch layers (Linear, LayerNorm, etc.) are initialized by default.
        # Custom nn.Parameter (like in CharacterDiacriticCompatibility) are initialized where defined.
        # This method can be expanded if specific non-default initializations are needed for custom modules.
        pass


    # --- Grad-CAM Hook Methods ---
    def _save_activation(self, module, input, output):
        self.activations = output
        if self.activations is not None and self.activations.requires_grad:
            self.activations.retain_grad()
            logger.debug(f"Called retain_grad() on activations from {module.__class__.__name__}")

    def _register_hooks(self, target_layer_module):
        self.clear_hooks()
        logger.debug(f"Registering forward hook for Grad-CAM on: {target_layer_module.__class__.__name__}")
        handle_f = target_layer_module.register_forward_hook(self._save_activation)
        self.activation_hook_handles.append(handle_f)

    def clear_hooks(self):
        for handle in self.activation_hook_handles: handle.remove()
        self.activation_hook_handles = []
        self.activations = None # Gradients will be cleared when activations are cleared

    def get_activations_and_gradients(self):
        current_gradients = None
        current_activations = None
        if self.activations is not None:
            current_activations = self.activations.detach().clone() # Detach and clone for safety
            if self.activations.grad is not None:
                current_gradients = self.activations.grad.detach().clone()
                logger.debug("Successfully retrieved .grad from hooked activations.")
            else:
                logger.warning("Gradients (.grad) for hooked activations were not populated.")
        else:
            logger.warning("Activations were None when trying to get gradients for Grad-CAM.")
        return current_activations, current_gradients
    # --- End Grad-CAM Hook Methods ---

    def forward(self, pixel_values, labels=None, label_lengths=None,
                return_diacritic_attention=False,
                grad_cam_target_layer_module_name=None):

        if grad_cam_target_layer_module_name and not pixel_values.requires_grad:
            pixel_values.requires_grad_(True) # Ensure input allows grad flow if needed for CAM

        # 1. Vision Encoder
        encoder_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        all_hidden_states = encoder_outputs.hidden_states

        # 2. Select Features for Fusion
        num_enc_layers = len(all_hidden_states)
        valid_indices = sorted(list(set(idx if idx >= 0 else num_enc_layers + idx for idx in self.config.vision_encoder_layer_indices))) # Handles negative indices relative to actual num_enc_layers
        valid_indices = [i for i in valid_indices if 0 <= i < num_enc_layers] # Ensure valid range
        if not valid_indices: features_to_fuse = [all_hidden_states[-1]]
        else: features_to_fuse = [all_hidden_states[i] for i in valid_indices]

        # 2b. Resolve and Register Grad-CAM Hook
        target_module_for_grad_cam = None
        if grad_cam_target_layer_module_name:
            # Simple attribute lookup first
            if hasattr(self, grad_cam_target_layer_module_name):
                target_module_for_grad_cam = getattr(self, grad_cam_target_layer_module_name)
            # Handle nested modules like transformer_encoder.layers.X
            elif '.' in grad_cam_target_layer_module_name:
                try:
                    current_obj = self
                    for part in grad_cam_target_layer_module_name.split('.'):
                        if part.isdigit(): current_obj = current_obj[int(part)] # For ModuleList
                        else: current_obj = getattr(current_obj, part)
                    target_module_for_grad_cam = current_obj
                except (AttributeError, IndexError, ValueError) as e:
                    logger.warning(f"Could not resolve nested Grad-CAM target '{grad_cam_target_layer_module_name}': {e}")
            
            if target_module_for_grad_cam is not None and isinstance(target_module_for_grad_cam, nn.Module):
                self._register_hooks(target_module_for_grad_cam)
            elif target_module_for_grad_cam is None:
                 logger.warning(f"Grad-CAM target layer '{grad_cam_target_layer_module_name}' not found or is not an nn.Module.")


        # 2c. Apply Fusion
        fused_features = None
        if self.dynamic_fusion: # This is an nn.Module, check if it's defined
            fused_features = self.dynamic_fusion(features_to_fuse)
        elif self.fusion_projection and self.config.feature_fusion_method == "concat_proj":
            fused_features = self.fusion_projection(torch.cat(features_to_fuse, dim=-1))
        elif self.pre_fusion_projections and self.config.feature_fusion_method == "add":
            fused_features = torch.stack([proj(feat) for proj, feat in zip(self.pre_fusion_projections, features_to_fuse)]).mean(0)
        elif self.fusion_bilinear: 
            fused_features = self.fusion_bilinear(features_to_fuse[0], features_to_fuse[1])
        elif self.fusion_projection: # Single layer projection
            fused_features = self.fusion_projection(features_to_fuse[-1]) 
        else: # Direct use of last selected layer if dimensions match, or fallback
             fused_features = features_to_fuse[-1]

        # Apply post_static_fusion_norm if it's defined (i.e., static fusion was configured)
        if self.post_static_fusion_norm is not None:
            fused_features = self.post_static_fusion_norm(fused_features)
        
        # 3. Feature Enhancer
        enhanced_features = self.feature_enhancer(fused_features) if self.feature_enhancer else fused_features
        # 4. Positional Encoding
        features_with_pos = self.pos_encoder(enhanced_features)
        # 5. Transformer Encoder
        transformer_output = self.transformer_encoder(features_with_pos)
        # 6. Shared Layer
        shared_features = self.shared_layer(transformer_output) # `self.activations` set here if "shared_layer" is target

        # 7. Hierarchical Heads
        base_logits = self.base_classifier(shared_features)
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat_proj' and self.diacritic_condition_proj:
            diacritic_input_features = self.diacritic_condition_proj(torch.cat((shared_features, F.softmax(base_logits, dim=-1)), dim=-1))
        elif self.config.conditioning_method == 'gate' and self.diacritic_gate:
            diacritic_input_features = shared_features * self.diacritic_gate(torch.cat((shared_features, F.softmax(base_logits, dim=-1)), dim=-1))

        standard_diacritic_logits = self.diacritic_classifier(diacritic_input_features)
        current_diacritic_logits = standard_diacritic_logits
        vda_raw_output, visual_diacritic_attention_maps = None, None
        compatibility_matrices = []
        
        if self.visual_diacritic_attention:
            res = self.visual_diacritic_attention(diacritic_input_features, return_attention_weights=True)
            vda_raw_output, visual_diacritic_attention_maps = res if isinstance(res, tuple) else (res, None)
            current_diacritic_logits = current_diacritic_logits + vda_raw_output
        if self.character_diacritic_compatibility:
            # Get compatibility bias and matrix
            compat_bias, compat_matrix = self.character_diacritic_compatibility(base_logits, diacritic_input_features)
            current_diacritic_logits = current_diacritic_logits + compat_bias
            compatibility_matrices.append(compat_matrix)  # Save for loss function

        
        if self.few_shot_diacritic_adapter: current_diacritic_logits = current_diacritic_logits + self.few_shot_diacritic_adapter(diacritic_input_features)
        diacritic_logits = current_diacritic_logits

        # 8. Final Classifier
        final_logits = self.final_classifier(shared_features)

        # 9. CTC Loss
        loss = None # Will hold the final combined loss if labels are provided
        
        # Initialize components of the loss
        ctc_loss_component = torch.tensor(0.0, device=pixel_values.device)
        reg_loss_component = torch.tensor(0.0, device=pixel_values.device)

        if labels is not None and label_lengths is not None:
            # A. Calculate CTC Loss for final_logits
            log_probs = final_logits.log_softmax(dim=2).permute(1, 0, 2) # (SeqLen, Batch, VocabSize)
            
            current_device = log_probs.device
            seq_len = log_probs.size(0)
            batch_size = log_probs.size(1)
            
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=current_device)
            
            labels_on_device = labels.to(current_device)
            label_lengths_on_device = label_lengths.to(current_device)
            
            # Clamp label lengths to be at most the size of the label tensor's second dimension
            # and also ensure they are not greater than input_lengths (seq_len)
            clamped_label_lengths = torch.clamp(label_lengths_on_device, min=0, max=labels_on_device.size(1)) # Ensure min=0
            clamped_label_lengths = torch.clamp(clamped_label_lengths, min=0, max=seq_len) # Ensure min=0


            # Ensure CTCLoss requires_grad if it's zero (e.g., no valid samples)
            # so that subsequent addition of reg_loss_component doesn't lose its grad requirement.
            # Initialize with requires_grad=True so that if no valid samples for CTC, this tensor can still accumulate gradients from reg_loss
            ctc_loss_component = torch.tensor(0.0, device=current_device, requires_grad=True) 
            
            try:
                ctc_loss_fn = nn.CTCLoss(blank=self.config.blank_idx, reduction='mean', zero_infinity=True)
                
                # Mask for valid samples (label length > 0)
                valid_samples_mask = clamped_label_lengths > 0
                
                if torch.any(valid_samples_mask):
                    # Select only valid entries for CTC loss
                    active_log_probs = log_probs[:, valid_samples_mask, :]
                    active_labels = labels_on_device[valid_samples_mask]
                    # Ensure input_lengths corresponds to the selected samples for B_active
                    active_input_lengths = input_lengths[valid_samples_mask] 
                    active_label_lengths = clamped_label_lengths[valid_samples_mask]

                    # Guard against label_length > input_length for selected active samples,
                    # though clamping label_lengths to seq_len should largely prevent this.
                    # This check might be redundant if clamping is perfect.
                    mask_len_ok = active_label_lengths <= active_input_lengths
                    if not torch.all(mask_len_ok):
                        logger.warning(f"Found {torch.sum(~mask_len_ok)} samples where label_length > input_length after initial clamping and masking. Filtering these for CTC.")
                        active_log_probs = active_log_probs[:, mask_len_ok, :]
                        active_labels = active_labels[mask_len_ok]
                        active_input_lengths = active_input_lengths[mask_len_ok]
                        active_label_lengths = active_label_lengths[mask_len_ok]
                    
                    if active_labels.numel() > 0 and active_log_probs.numel() > 0 : # If any samples remain after filtering
                        # Further check: if log_probs themselves are NaN/Inf, ctc_loss_fn will also be.
                        if torch.isnan(active_log_probs).any() or torch.isinf(active_log_probs).any():
                            logger.warning("Log_probs for CTC loss contained NaN or Inf. Skipping CTC calculation for this batch.")
                            # ctc_loss_component remains 0.0 with requires_grad=True
                        else:
                            calculated_ctc_loss = ctc_loss_fn(
                                active_log_probs,
                                active_labels,
                                active_input_lengths,
                                active_label_lengths
                            )
                            if not (torch.isnan(calculated_ctc_loss) or torch.isinf(calculated_ctc_loss)):
                                ctc_loss_component = calculated_ctc_loss
                            else:
                                logger.warning("CTC loss resulted in NaN or Inf. Using 0.0 for this component.")
                                # ctc_loss_component remains 0.0 with requires_grad=True
                # If no valid samples, ctc_loss_component remains 0.0 with requires_grad=True
            except Exception as e:
                logger.error(f"CTC loss calculation error: {e}", exc_info=True)
                # ctc_loss_component remains 0.0 with requires_grad=True

            # B. Calculate Compatibility Regularization Loss
            if hasattr(self, 'character_diacritic_compatibility') and \
               self.character_diacritic_compatibility is not None and \
               self.config.use_character_diacritic_compatibility: 
                
                compat_matrix = self.character_diacritic_compatibility.compatibility_matrix
                
                if torch.isnan(compat_matrix).any() or torch.isinf(compat_matrix).any():
                    logger.warning("Compatibility matrix contains NaN/Inf. Skipping regularization loss calculation.")
                    reg_loss_component = torch.tensor(0.0, device=current_device, requires_grad=True)
                else:
                    mean_val = compat_matrix.mean()
                    target_mean = torch.tensor(-0.1, device=mean_val.device)
                    loss_mean_reg = F.mse_loss(mean_val, target_mean)
                    
                    variance_val = torch.var(compat_matrix, unbiased=False) 
                    target_variance = torch.tensor(1.0, device=variance_val.device)
                    loss_var_reg = F.mse_loss(variance_val, target_variance)
                    
                    current_reg_loss = 0.001 * (loss_mean_reg + loss_var_reg)
                    if not (torch.isnan(current_reg_loss) or torch.isinf(current_reg_loss)):
                        reg_loss_component = current_reg_loss
                    else:
                        logger.warning("Regularization loss resulted in NaN or Inf. Using 0.0 for this component.")
                        reg_loss_component = torch.tensor(0.0, device=current_device, requires_grad=True)
            
            # C. Combine losses
            # Ensure loss doesn't become NaN if one component is NaN and the other is 0.0 with requires_grad=True
            if torch.isnan(ctc_loss_component) or torch.isnan(reg_loss_component):
                logger.error(f"NaN detected in loss components before summation! CTC: {ctc_loss_component}, Reg: {reg_loss_component}. Setting total loss to a fresh 0.0 tensor to prevent propagation.")
                loss = torch.tensor(0.0, device=current_device, requires_grad=True) # Fallback to prevent NaN propagation
            else:
                loss = ctc_loss_component + reg_loss_component
        
        # Populate output dictionary
        output_dict = {
            'loss': loss, 
            'logits': final_logits, 
            'base_logits': base_logits, 
            'diacritic_logits': diacritic_logits
        }
        if return_diacritic_attention and visual_diacritic_attention_maps is not None:
            output_dict['visual_diacritic_attention_maps'] = visual_diacritic_attention_maps
        
        if grad_cam_target_layer_module_name:
            grad_cam_target_logits = diacritic_logits # Default target for CAM on diacritics
            if self.visual_diacritic_attention and vda_raw_output is not None:
                 # If VDA is active and produced output, it's a more specific target
                grad_cam_target_logits = vda_raw_output
            output_dict['grad_cam_target_logits'] = grad_cam_target_logits
            
        return output_dict

    def save_pretrained(self, save_directory, **kwargs):
        logger.info(f"Saving {self.__class__.__name__} model to: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        # Ensure vocabs are in config
        self.config.base_char_vocab = self.base_char_vocab
        self.config.diacritic_vocab = self.diacritic_vocab
        self.config.combined_char_vocab = self.combined_char_vocab
        self.config.base_char_vocab_size = len(self.base_char_vocab)
        self.config.diacritic_vocab_size = len(self.diacritic_vocab)
        self.config.combined_char_vocab_size = len(self.combined_char_vocab)
        # Save vision_encoder_config as dict for better compatibility
        if hasattr(self.vision_encoder, 'config') and self.vision_encoder.config is not None:
            self.config.vision_encoder_config = self.vision_encoder.config.to_dict()
        else:
            logger.warning("Vision encoder config not found or is None, cannot save to dict.")
            self.config.vision_encoder_config = {} # Save empty dict

        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        if hasattr(self, 'processor') and self.processor:
            try: self.processor.save_pretrained(save_directory)
            except Exception as e: logger.warning(f"Could not save processor: {e}")
        logger.info("Model components saved.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        logger.info(f"Loading {cls.__name__} from: {pretrained_model_name_or_path}")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        loaded_config_dict = {} # For kwargs passed to config init

        if config is not None and isinstance(config, cls.config_class):
            loaded_config = config
            logger.info("Using provided config object.")
            # Update with kwargs
            for key, value in kwargs.items():
                if hasattr(loaded_config, key): setattr(loaded_config, key, value)
                else: loaded_config_dict[key] = value # Store for later if it's a new config arg
        elif os.path.exists(config_path):
            # from_pretrained on config class will handle kwargs override
            loaded_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
            logger.info(f"Loaded config from file, applied/overrode with kwargs: {kwargs}")
        else: # No config file, attempt to init from kwargs or defaults
            logger.warning(f"Config file not found at {config_path}. Initializing new config.")
            # Pass all kwargs to config constructor
            loaded_config_dict.update(kwargs)
            # Ensure essential vocabs are present if not in defaults
            if 'combined_char_vocab' not in loaded_config_dict:
                raise ValueError("combined_char_vocab is required if config.json is not present.")
            loaded_config = cls.config_class(**loaded_config_dict)

        # Ensure vision_encoder_config is an object
        if hasattr(loaded_config, 'vision_encoder_config') and isinstance(loaded_config.vision_encoder_config, dict):
            vision_config_data = loaded_config.vision_encoder_config
            try: # Try to get specific config class
                from transformers import AutoConfig
                # Use vision_encoder_name from the config to get a hint for AutoConfig
                # Default to vision_encoder_name itself if model_type isn't in vision_config_data
                vision_model_identifier = vision_config_data.get("model_type", loaded_config.vision_encoder_name)

                if "trocr" in loaded_config.vision_encoder_name.lower() and "vit" not in vision_model_identifier.lower() : # TrOCR special case
                    from transformers import ViTConfig
                    vision_config_obj = ViTConfig(**vision_config_data)
                    logger.info("Using ViTConfig for TrOCR-like model based on vision_encoder_name.")
                else:
                    # Attempt to get the specific config class using AutoConfig
                    # This relies on 'model_type' in vision_config_data or vision_encoder_name being informative
                    specific_vision_config = AutoConfig.from_pretrained(vision_model_identifier, trust_remote_code=True)
                    vision_config_obj = specific_vision_config.__class__(**vision_config_data)

                loaded_config.vision_encoder_config = vision_config_obj
                logger.info(f"Converted vision_encoder_config dict to {vision_config_obj.__class__.__name__} object.")
            except Exception as e_conf:
                logger.warning(f"Could not auto-detect specific vision config type for '{vision_model_identifier}': {e_conf}. Using PretrainedConfig as base.")
                base_vision_conf = PretrainedConfig() # Fallback
                for k_vc, v_vc in vision_config_data.items(): setattr(base_vision_conf, k_vc, v_vc)
                loaded_config.vision_encoder_config = base_vision_conf
        
        # Update vocab lists and sizes from kwargs if they were explicitly passed
        for vocab_name in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']:
            if vocab_name in kwargs and kwargs[vocab_name]:
                setattr(loaded_config, vocab_name, kwargs[vocab_name])
                setattr(loaded_config, f"{vocab_name}_size", len(kwargs[vocab_name]))

        model = cls(loaded_config) # Instantiate model with the finalized config

        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path) and os.path.exists(config_path): # Only load if it's a full checkpoint
            logger.info(f"Loading state dict from: {state_dict_path}")
            try:
                state_dict = torch.load(state_dict_path, map_location="cpu")
                load_result = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded state. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
            except Exception as e: logger.error(f"Error loading state dict: {e}", exc_info=True)
        else:
            logger.info(f"Not loading full model state_dict (either config.json or pytorch_model.bin missing at {pretrained_model_name_or_path}). Using base vision weights and/or random init for custom parts.")

        model.base_char_vocab = loaded_config.base_char_vocab
        model.diacritic_vocab = loaded_config.diacritic_vocab
        model.combined_char_vocab = loaded_config.combined_char_vocab
        return model

# --- Combined Multi-Scale Model Class ---
class HierarchicalCtcMultiScaleOcrModel(HierarchicalCtcTransformerOcrModel):
    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        logger.info(f"Initialized {self.__class__.__name__} (inherits Grad-CAM from parent).")