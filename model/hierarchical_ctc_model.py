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
from .diacritic_attention import (
    VisualDiacriticAttention,
    CharacterDiacriticCompatibility,
    FewShotDiacriticAdapter
)
from .dynamic_fusion import (
    DynamicMultiScaleFusion,
    LocalFeatureEnhancer
)
from datetime import datetime
logger = logging.getLogger(__name__)

# --- Configuration Class (HierarchicalCtcOcrConfig) ---
class HierarchicalCtcOcrConfig(PretrainedConfig):
    model_type = "hierarchical_ctc_transformer"

    def __init__(
        self,
        vision_encoder_name='microsoft/trocr-base-handwritten',
        base_char_vocab=None, diacritic_vocab=None, combined_char_vocab=None,
        vision_encoder_layer_indices=[-1, -4, -7], feature_fusion_method="concat_proj",
        use_dynamic_fusion=False, use_feature_enhancer=False,
        num_transformer_encoder_layers=4, transformer_d_model=512,
        transformer_nhead=8, transformer_dim_feedforward=2048,
        transformer_dropout=0.1, positional_encoding_type="sinusoidal_1d",
        max_seq_len_for_pos_enc=768, shared_hidden_size=512,
        num_shared_layers=1, conditioning_method="concat_proj",
        classifier_dropout=0.3, blank_idx=0, vision_encoder_config=None,
        use_visual_diacritic_attention=False, use_character_diacritic_compatibility=False,
        use_few_shot_diacritic_adapter=False, num_few_shot_prototypes=10,
        hierarchical_mode="sequential", # "parallel", "sequential", "multitask", "enhanced_single"
        multitask_loss_weights=None, #weights for multitask losses
        use_middle_diacritic_conditioning=True, 
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
        self.hierarchical_mode = hierarchical_mode
        # Default multitask loss weights [final_weight, base_weight, diacritic_weight]
        if multitask_loss_weights is None:
            self.multitask_loss_weights = [1.0, 0.1, 0.1]
        else:
            self.multitask_loss_weights = multitask_loss_weights
        self.use_middle_diacritic_conditioning = use_middle_diacritic_conditioning
        # Validate hierarchical mode
        valid_modes = ["parallel", "sequential", "multitask", "enhanced_single"]
        if self.hierarchical_mode not in valid_modes:
            raise ValueError(f"Invalid hierarchical_mode: {self.hierarchical_mode}. Must be one of {valid_modes}")

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
        # self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
        # logger.info(f"Base Classifier (In: {shared_output_size}, Out: {config.base_char_vocab_size})")

        # self.diacritic_gate = None; self.diacritic_condition_proj = None
        # diacritic_head_input_size = shared_output_size
        # if config.conditioning_method == 'concat_proj':
        #     self.diacritic_condition_proj = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.LayerNorm(shared_output_size), nn.GELU(), nn.Dropout(config.classifier_dropout))
        #     logger.info(f"'concat_proj' conditioning for diacritic head -> {shared_output_size} dim.")
        # elif config.conditioning_method == 'gate':
        #     self.diacritic_gate = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.Sigmoid())
        #     logger.info(f"'gate' conditioning for diacritic head -> {shared_output_size} dim.")
        # else: logger.info(f"'none' conditioning for diacritic head -> {shared_output_size} dim.")

        # self.visual_diacritic_attention = None; self.character_diacritic_compatibility = None; self.few_shot_diacritic_adapter = None
        # if config.use_visual_diacritic_attention: self.visual_diacritic_attention = VisualDiacriticAttention(diacritic_head_input_size, config.diacritic_vocab_size); logger.info("Visual Diacritic Attn: ON")
        # if config.use_character_diacritic_compatibility: self.character_diacritic_compatibility = CharacterDiacriticCompatibility(config.base_char_vocab_size, config.diacritic_vocab_size, diacritic_head_input_size, self.base_char_vocab, self.diacritic_vocab); logger.info("Char-Diac Compat: ON")
        # if config.use_few_shot_diacritic_adapter: self.few_shot_diacritic_adapter = FewShotDiacriticAdapter(diacritic_head_input_size, config.diacritic_vocab_size, config.num_few_shot_prototypes); logger.info("Few-Shot Adapter: ON")

        # self.diacritic_classifier = nn.Linear(diacritic_head_input_size, config.diacritic_vocab_size)
        # logger.info(f"Diacritic Classifier (In: {diacritic_head_input_size}, Out: {config.diacritic_vocab_size})")

        # self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)
        # logger.info(f"Final Combined Classifier (In: {shared_output_size}, Out: {config.combined_char_vocab_size})")

        # --- Adaptive Hierarchical Classification Heads ---
        self._setup_classification_heads(config, shared_output_size)
        logger.info(f"Adaptive Hierarchical Classification Heads set up with {config.hierarchical_mode} mode.")
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

    def _setup_classification_heads(self, config, shared_output_size):
        """Setup classification heads based on hierarchical mode"""
        
        if config.hierarchical_mode == "sequential":
            self._setup_sequential_heads(config, shared_output_size)
        elif config.hierarchical_mode == "multitask":
            self._setup_multitask_heads(config, shared_output_size)
        elif config.hierarchical_mode == "parallel":
            self._setup_parallel_heads(config, shared_output_size)
        else:  # enhanced_single
            self._setup_enhanced_single_heads(config, shared_output_size)

    def _setup_sequential_heads(self, config, shared_output_size):
        """Sequential hierarchical: base → diacritic → final"""
        logger.info("Setting up SEQUENTIAL hierarchical classification heads")
        
        # Base classifier
        self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
        logger.info(f"Base Classifier (In: {shared_output_size}, Out: {config.base_char_vocab_size})")

        # Diacritic classifier with base conditioning
        diacritic_input_size = shared_output_size + config.base_char_vocab_size
        self.diacritic_fusion = nn.Sequential(
            nn.Linear(diacritic_input_size, shared_output_size),
            nn.LayerNorm(shared_output_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout)
        )
        
        self._setup_diacritic_enhancements(config, shared_output_size)
        self.diacritic_classifier = nn.Linear(shared_output_size, config.diacritic_vocab_size)
        logger.info(f"Diacritic Classifier (In: {shared_output_size}, Out: {config.diacritic_vocab_size})")

        # Final classifier with both base and diacritic conditioning
        final_input_size = shared_output_size + config.base_char_vocab_size + config.diacritic_vocab_size
        self.final_fusion = nn.Sequential(
            nn.Linear(final_input_size, shared_output_size),
            nn.LayerNorm(shared_output_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout)
        )
        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)
        logger.info(f"Final Classifier (In: {shared_output_size}, Out: {config.combined_char_vocab_size}) - Sequential")

    def _setup_multitask_heads(self, config, shared_output_size):
        """Multi-task learning: three parallel tasks with shared features"""
        logger.info("Setting up MULTITASK parallel classification heads")
        
        # All three classifiers operate on shared features
        self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
        logger.info(f"Base Classifier (In: {shared_output_size}, Out: {config.base_char_vocab_size})")

        # Diacritic classifier with optional conditioning
        if config.conditioning_method != 'none':
            self._setup_diacritic_conditioning(config, shared_output_size)
        
        self._setup_diacritic_enhancements(config, shared_output_size)
        self.diacritic_classifier = nn.Linear(shared_output_size, config.diacritic_vocab_size)
        logger.info(f"Diacritic Classifier (In: {shared_output_size}, Out: {config.diacritic_vocab_size})")

        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)
        logger.info(f"Final Classifier (In: {shared_output_size}, Out: {config.combined_char_vocab_size}) - Multitask")

    def _setup_parallel_heads(self, config, shared_output_size):
        """Original parallel heads (existing implementation)"""
        logger.info("Setting up PARALLEL classification heads (original)")
        
        # Keep existing implementation
        self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
        
        if config.conditioning_method != 'none':
            self._setup_diacritic_conditioning(config, shared_output_size)
        
        self._setup_diacritic_enhancements(config, shared_output_size)
        self.diacritic_classifier = nn.Linear(shared_output_size, config.diacritic_vocab_size)
        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)

    def _setup_enhanced_single_heads(self, config, shared_output_size):
        """Enhanced single task: only final classifier with enhancements"""
        logger.info("Setting up ENHANCED SINGLE classification head")
        
        # Only final classifier, but with enhancements applied directly
        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size)
        
        # Setup enhancements to be applied to final classifier
        self._setup_diacritic_enhancements(config, shared_output_size)
        
        # Optional: keep base classifier for compatibility matrix
        if config.use_character_diacritic_compatibility:
            self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size)
            logger.info("Added base classifier for compatibility matrix support")

    def _setup_diacritic_conditioning(self, config, shared_output_size):
        """Setup diacritic conditioning layers"""
        if config.conditioning_method == 'concat_proj':
            self.diacritic_condition_proj = nn.Sequential(
                nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size),
                nn.LayerNorm(shared_output_size),
                nn.GELU(),
                nn.Dropout(config.classifier_dropout)
            )
            logger.info("Diacritic conditioning: concat_proj")
        elif config.conditioning_method == 'gate':
            self.diacritic_gate = nn.Sequential(
                nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size),
                nn.Sigmoid()
            )
            logger.info("Diacritic conditioning: gate")

    def _setup_diacritic_enhancements(self, config, shared_output_size):
        """Setup diacritic enhancement modules"""
        self.visual_diacritic_attention = None
        self.character_diacritic_compatibility = None
        self.few_shot_diacritic_adapter = None
        
        if config.use_visual_diacritic_attention:
            self.visual_diacritic_attention = VisualDiacriticAttention(shared_output_size, config.diacritic_vocab_size)
            logger.info("Visual Diacritic Attention: ON")
            
        if config.use_character_diacritic_compatibility:
            self.character_diacritic_compatibility = CharacterDiacriticCompatibility(
                config.base_char_vocab_size, config.diacritic_vocab_size, 
                shared_output_size, self.base_char_vocab, self.diacritic_vocab
            )
            logger.info("Character-Diacritic Compatibility: ON")
            
        if config.use_few_shot_diacritic_adapter:
            self.few_shot_diacritic_adapter = FewShotDiacriticAdapter(
                shared_output_size, config.diacritic_vocab_size, config.num_few_shot_prototypes
            )
            logger.info("Few-Shot Diacritic Adapter: ON")

    def forward(self, pixel_values, labels=None, label_lengths=None,
                return_diacritic_attention=False, grad_cam_target_layer_module_name=None):

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
        # Route to appropriate forward method based on mode
        if self.config.hierarchical_mode == "sequential":
            return self._forward_sequential(shared_features, labels, label_lengths, 
                                          return_diacritic_attention, grad_cam_target_layer_module_name)
        elif self.config.hierarchical_mode == "multitask":
            return self._forward_multitask(shared_features, labels, label_lengths,
                                         return_diacritic_attention, grad_cam_target_layer_module_name)
        elif self.config.hierarchical_mode == "enhanced_single":
            return self._forward_enhanced_single(shared_features, labels, label_lengths,
                                               return_diacritic_attention, grad_cam_target_layer_module_name)
        else:  # parallel (original)
            return self._forward_parallel(shared_features, labels, label_lengths,
                                        return_diacritic_attention, grad_cam_target_layer_module_name)

    def _forward_sequential(self, shared_features, labels=None, label_lengths=None,
                           return_diacritic_attention=False, grad_cam_target_layer_module_name=None):
        """Sequential hierarchical: base → diacritic → final"""
        
        # Step 1: Base character prediction
        base_logits = self.base_classifier(shared_features)
        base_probs = F.softmax(base_logits, dim=-1)
        # 🔍 DEBUG: Log base predictions
        if self.training and torch.rand(1).item() < 0.001:  # Log 0.1% of batches
            self._debug_log_base_predictions(base_logits, base_probs)
        # Step 2: Diacritic prediction conditioned on base
        if self.config.use_middle_diacritic_conditioning:
            # Original: Condition diacritic on base predictions
            base_enhanced_features = torch.cat([shared_features, base_probs], dim=-1)
            diacritic_features = self.diacritic_fusion(base_enhanced_features)
            logger.debug("Using middle diacritic conditioning (baseline)")
        else:
            # Ablation: Skip middle conditioning, use shared features directly
            diacritic_features = shared_features
            logger.debug("Ablation: Skipping middle diacritic conditioning")
        # Get raw diacritic logits BEFORE compatibility
        raw_diacritic_logits = self.diacritic_classifier(diacritic_features)

        # Apply diacritic enhancements
        enhanced_diacritic_logits, attention_maps = self._apply_diacritic_enhancements(
            diacritic_features, base_logits, return_diacritic_attention
        )
        diacritic_probs = F.softmax(enhanced_diacritic_logits, dim=-1)
        # 🔍 DEBUG: Log compatibility effects
        if self.training and torch.rand(1).item() < 0.001:
            self._debug_log_compatibility_effects(
                base_logits, raw_diacritic_logits, enhanced_diacritic_logits
            )
        # Step 3: Final prediction conditioned on both
        diacritic_probs = F.softmax(enhanced_diacritic_logits, dim=-1)
        final_enhanced_features = torch.cat([shared_features, base_probs, diacritic_probs], dim=-1)
        final_features = self.final_fusion(final_enhanced_features)
        final_logits = self.final_classifier(final_features)
        
        # Compute loss
        loss = None
        if labels is not None and label_lengths is not None:
            loss = self._compute_loss(final_logits, base_logits, enhanced_diacritic_logits, 
                                    labels, label_lengths, mode="sequential")
        
        return self._prepare_output(final_logits, base_logits, enhanced_diacritic_logits, loss,
                                  attention_maps, return_diacritic_attention,
                                  grad_cam_target_layer_module_name)
    
    def _debug_log_base_predictions(self, base_logits, base_probs):
        """Log top base character predictions"""
        batch_size = base_logits.shape[0]
        for b in range(min(2, batch_size)):  # Log first 2 samples
            for t in range(min(5, base_logits.shape[1])):  # Log first 5 timesteps
                top_indices = torch.topk(base_probs[b, t], k=3).indices
                top_chars = [self.base_char_vocab[idx] for idx in top_indices]
                top_probs = [base_probs[b, t, idx].item() for idx in top_indices]
                
                logger.info(f"Batch {b}, Time {t} - Top base predictions: {list(zip(top_chars, top_probs))}")

    def _debug_log_compatibility_effects(self, base_logits, raw_diacritic_logits, enhanced_diacritic_logits):
        """Log how compatibility matrix affects diacritic predictions"""
        # Calculate the compatibility bias
        compatibility_bias = enhanced_diacritic_logits - raw_diacritic_logits
        
        batch_size = base_logits.shape[0]
        for b in range(min(2, batch_size)):
            for t in range(min(3, base_logits.shape[1])):
                # Get top base character
                top_base_idx = torch.argmax(base_logits[b, t]).item()
                top_base_char = self.base_char_vocab[top_base_idx]
                
                # Get diacritic changes
                bias = compatibility_bias[b, t]
                top_positive_bias = torch.topk(bias, k=3)
                top_negative_bias = torch.topk(bias, k=3, largest=False)
                
                pos_diacritics = [self.diacritic_vocab[idx] for idx in top_positive_bias.indices]
                neg_diacritics = [self.diacritic_vocab[idx] for idx in top_negative_bias.indices]
                
                logger.info(f"Base '{top_base_char}' → Boosted diacritics: {pos_diacritics}")
                logger.info(f"Base '{top_base_char}' → Suppressed diacritics: {neg_diacritics}")

    def _forward_multitask(self, shared_features, labels=None, label_lengths=None,
                          return_diacritic_attention=False, grad_cam_target_layer_module_name=None):
        """Multi-task learning: three parallel tasks"""
        
        # All predictions from shared features
        base_logits = self.base_classifier(shared_features)
        
        # Diacritic prediction with optional conditioning
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat_proj' and hasattr(self, 'diacritic_condition_proj'):
            base_probs = F.softmax(base_logits, dim=-1)
            diacritic_input_features = self.diacritic_condition_proj(
                torch.cat([shared_features, base_probs], dim=-1)
            )
        elif self.config.conditioning_method == 'gate' and hasattr(self, 'diacritic_gate'):
            base_probs = F.softmax(base_logits, dim=-1)
            gate = self.diacritic_gate(torch.cat([shared_features, base_probs], dim=-1))
            diacritic_input_features = shared_features * gate
        
        # Apply diacritic enhancements
        diacritic_logits, visual_diacritic_attention_maps = self._apply_diacritic_enhancements(
            diacritic_input_features, base_logits, return_diacritic_attention
        )
        
        # Final prediction
        final_logits = self.final_classifier(shared_features)
        
        # Compute multi-task loss
        loss = None
        if labels is not None and label_lengths is not None:
            loss = self._compute_loss(final_logits, base_logits, diacritic_logits,
                                    labels, label_lengths, mode="multitask")
        
        return self._prepare_output(final_logits, base_logits, diacritic_logits, loss,
                                  visual_diacritic_attention_maps, return_diacritic_attention,
                                  grad_cam_target_layer_module_name)

    def _forward_enhanced_single(self, shared_features, labels=None, label_lengths=None,
                                return_diacritic_attention=False, grad_cam_target_layer_module_name=None):
        """Enhanced single task: apply enhancements directly to final classifier"""
        
        # Base prediction for compatibility matrix (if needed)
        base_logits = None
        if hasattr(self, 'base_classifier'):
            base_logits = self.base_classifier(shared_features)
        
        # Final prediction with enhancements
        final_logits = self.final_classifier(shared_features)
        
        # Apply enhancements directly to final logits
        visual_diacritic_attention_maps = None
        
        if self.visual_diacritic_attention:
            if return_diacritic_attention:
                vda_output, visual_diacritic_attention_maps = self.visual_diacritic_attention(
                    shared_features, return_attention_weights=True
                )
            else:
                vda_output = self.visual_diacritic_attention(shared_features)
            final_logits = final_logits + vda_output
        
        if self.character_diacritic_compatibility and base_logits is not None:
            compat_bias, _ = self.character_diacritic_compatibility(base_logits, shared_features)
            final_logits = final_logits + compat_bias
        
        if self.few_shot_diacritic_adapter:
            fsa_output = self.few_shot_diacritic_adapter(shared_features)
            final_logits = final_logits + fsa_output
        
        # Dummy diacritic logits for compatibility
        diacritic_logits = torch.zeros(final_logits.shape[0], final_logits.shape[1], 
                                     self.config.diacritic_vocab_size, device=final_logits.device)
        
        # Compute loss
        loss = None
        if labels is not None and label_lengths is not None:
            loss = self._compute_loss(final_logits, base_logits, diacritic_logits,
                                    labels, label_lengths, mode="enhanced_single")
        
        return self._prepare_output(final_logits, base_logits, diacritic_logits, loss,
                                  visual_diacritic_attention_maps, return_diacritic_attention,
                                  grad_cam_target_layer_module_name)

    def _forward_parallel(self, shared_features, labels=None, label_lengths=None,
                         return_diacritic_attention=False, grad_cam_target_layer_module_name=None):
        """Original parallel implementation"""
        
        # Keep existing parallel implementation
        base_logits = self.base_classifier(shared_features)
        
        # Existing diacritic processing...
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat_proj' and hasattr(self, 'diacritic_condition_proj'):
            base_probs = F.softmax(base_logits, dim=-1)
            diacritic_input_features = self.diacritic_condition_proj(
                torch.cat([shared_features, base_probs], dim=-1)
            )
        elif self.config.conditioning_method == 'gate' and hasattr(self, 'diacritic_gate'):
            base_probs = F.softmax(base_logits, dim=-1)
            gate = self.diacritic_gate(torch.cat([shared_features, base_probs], dim=-1))
            diacritic_input_features = shared_features * gate
        
        diacritic_logits, visual_diacritic_attention_maps = self._apply_diacritic_enhancements(
            diacritic_input_features, base_logits, return_diacritic_attention
        )
        
        final_logits = self.final_classifier(shared_features)
        
        loss = None
        if labels is not None and label_lengths is not None:
            loss = self._compute_loss(final_logits, base_logits, diacritic_logits,
                                    labels, label_lengths, mode="parallel")
        
        return self._prepare_output(final_logits, base_logits, diacritic_logits, loss,
                                  visual_diacritic_attention_maps, return_diacritic_attention,
                                  grad_cam_target_layer_module_name)

    def _apply_diacritic_enhancements(self, diacritic_features, base_logits, return_attention_weights=False):
        """Apply diacritic enhancement modules"""
        
        standard_diacritic_logits = self.diacritic_classifier(diacritic_features)
        enhanced_logits = standard_diacritic_logits
        visual_attention_maps = None
        
        if self.visual_diacritic_attention:
            if return_attention_weights:
                vda_output, visual_attention_maps = self.visual_diacritic_attention(
                    diacritic_features, return_attention_weights=True
                )
            else:
                vda_output = self.visual_diacritic_attention(diacritic_features)
            enhanced_logits = enhanced_logits + vda_output
        
        if self.character_diacritic_compatibility and base_logits is not None:
            compat_bias, _ = self.character_diacritic_compatibility(base_logits, diacritic_features)
            enhanced_logits = enhanced_logits + compat_bias
        
        if self.few_shot_diacritic_adapter:
            fsa_output = self.few_shot_diacritic_adapter(diacritic_features)
            enhanced_logits = enhanced_logits + fsa_output
        
        return enhanced_logits, visual_attention_maps

    def _compute_loss(self, final_logits, base_logits, diacritic_logits, labels, label_lengths, mode):
        """Compute loss based on hierarchical mode"""
        
        device = final_logits.device
        
        if mode == "multitask":
            # Multi-task loss: weighted combination of all three losses
            final_loss = self._compute_ctc_loss(final_logits, labels, label_lengths)
            
            # For multi-task, we need separate labels for base and diacritic
            # For now, we'll only use final loss, but this can be extended
            base_loss = torch.tensor(0.0, device=device, requires_grad=True)
            diacritic_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # TODO: Implement separate label generation for base and diacritic tasks
            # This would require decomposing the combined labels
            
            weights = self.config.multitask_loss_weights
            total_loss = (weights[0] * final_loss + 
                         weights[1] * base_loss + 
                         weights[2] * diacritic_loss)
            
        elif mode == "sequential":
            # Sequential: only final loss matters (intermediate predictions are steps)
            total_loss = self._compute_ctc_loss(final_logits, labels, label_lengths)
            
        else:  # parallel, enhanced_single
            # Single task: only final loss
            total_loss = self._compute_ctc_loss(final_logits, labels, label_lengths)
        
        # Add regularization losses
        reg_loss = self._compute_regularization_loss(device)
        total_loss = total_loss + reg_loss
        
        return total_loss

    def _compute_ctc_loss(self, logits, labels, label_lengths):
        """Helper method to compute CTC loss (existing implementation)"""
        # ... existing CTC loss computation code ...
        log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
        device = log_probs.device
        seq_len = log_probs.size(0)
        batch_size = log_probs.size(1)
        
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        labels_on_device = labels.to(device)
        label_lengths_on_device = label_lengths.to(device)
        
        clamped_label_lengths = torch.clamp(label_lengths_on_device, min=0, max=labels_on_device.size(1))
        clamped_label_lengths = torch.clamp(clamped_label_lengths, min=0, max=seq_len)
        
        ctc_loss_component = torch.tensor(0.0, device=device, requires_grad=True)
        
        try:
            ctc_loss_fn = nn.CTCLoss(blank=self.config.blank_idx, reduction='mean', zero_infinity=True)
            valid_samples_mask = clamped_label_lengths > 0
            
            if torch.any(valid_samples_mask):
                active_log_probs = log_probs[:, valid_samples_mask, :]
                active_labels = labels_on_device[valid_samples_mask]
                active_input_lengths = input_lengths[valid_samples_mask]
                active_label_lengths = clamped_label_lengths[valid_samples_mask]
                
                mask_len_ok = active_label_lengths <= active_input_lengths
                if not torch.all(mask_len_ok):
                    logger.warning(f"Filtering {torch.sum(~mask_len_ok)} samples where label_length > input_length")
                    active_log_probs = active_log_probs[:, mask_len_ok, :]
                    active_labels = active_labels[mask_len_ok]
                    active_input_lengths = active_input_lengths[mask_len_ok]
                    active_label_lengths = active_label_lengths[mask_len_ok]
                
                if active_labels.numel() > 0 and active_log_probs.numel() > 0:
                    if not (torch.isnan(active_log_probs).any() or torch.isinf(active_log_probs).any()):
                        calculated_ctc_loss = ctc_loss_fn(
                            active_log_probs, active_labels, active_input_lengths, active_label_lengths
                        )
                        if not (torch.isnan(calculated_ctc_loss) or torch.isinf(calculated_ctc_loss)):
                            ctc_loss_component = calculated_ctc_loss
                        else:
                            logger.warning("CTC loss resulted in NaN or Inf")
                    else:
                        logger.warning("Log_probs contained NaN or Inf")
        except Exception as e:
            logger.error(f"CTC loss calculation error: {e}")
        
        return ctc_loss_component

    def _compute_regularization_loss(self, device):
        """Compute regularization losses including linguistic constraints"""
        reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Existing compatibility matrix regularization (keep this)
        if (hasattr(self, 'character_diacritic_compatibility') and 
            self.character_diacritic_compatibility is not None and
            self.config.use_character_diacritic_compatibility):
            
            compat_matrix = self.character_diacritic_compatibility.compatibility_matrix
            if not (torch.isnan(compat_matrix).any() or torch.isinf(compat_matrix).any()):
                # Original regularization (for variance and mean)
                mean_val = compat_matrix.mean()
                target_mean = torch.tensor(-0.1, device=mean_val.device)
                loss_mean_reg = F.mse_loss(mean_val, target_mean)
                
                variance_val = torch.var(compat_matrix, unbiased=False)
                target_variance = torch.tensor(1.0, device=variance_val.device)
                loss_var_reg = F.mse_loss(variance_val, target_variance)
                
                # 🔥 NEW: Linguistic regularization (stronger weight)
                linguistic_loss = self.character_diacritic_compatibility.get_linguistic_regularization_loss(
                    strength=0.1  # Adjust this weight - higher = stronger linguistic enforcement
                )
                
                # Combine all compatibility losses
                current_reg_loss = 0.001 * (loss_mean_reg + loss_var_reg) + linguistic_loss
                
                if not (torch.isnan(current_reg_loss) or torch.isinf(current_reg_loss)):
                    reg_loss = current_reg_loss
                else:
                    logger.warning("Regularization loss resulted in NaN or Inf")
        
        return reg_loss

    def _prepare_output(self, final_logits, base_logits, diacritic_logits, loss,
                       visual_attention_maps, return_diacritic_attention, grad_cam_target):
        """Prepare model output dictionary"""
        
        output_dict = {
            'loss': loss,
            'logits': final_logits,
            'base_logits': base_logits if base_logits is not None else torch.empty(0),
            'diacritic_logits': diacritic_logits
        }
        
        if return_diacritic_attention and visual_attention_maps is not None:
            output_dict['visual_diacritic_attention_maps'] = visual_attention_maps
        
        if grad_cam_target:
            # Choose appropriate target based on mode and available outputs
            if self.visual_diacritic_attention and diacritic_logits.numel() > 0:
                output_dict['grad_cam_target_logits'] = diacritic_logits
            elif final_logits.numel() > 0:
                output_dict['grad_cam_target_logits'] = final_logits
            else:
                output_dict['grad_cam_target_logits'] = None
        
        return output_dict

    def _compute_multitask_loss_with_decomposition(self, final_logits, base_logits, diacritic_logits, 
                                                labels, label_lengths):
        """Compute true multi-task loss with label decomposition"""
        
        device = final_logits.device
        
        # Decompose labels for multi-task learning
        try:
            from utils.label_decomposition import decompose_labels_for_multitask
            base_labels, diacritic_labels = decompose_labels_for_multitask(
                labels, self.combined_char_vocab, self.base_char_vocab, self.diacritic_vocab
            )
        except ImportError:
            logger.warning("Label decomposition not available, using single-task loss")
            return self._compute_ctc_loss(final_logits, labels, label_lengths)
        
        # Compute individual losses
        final_loss = self._compute_ctc_loss(final_logits, labels, label_lengths)
        base_loss = self._compute_ctc_loss(base_logits, base_labels, label_lengths)
        diacritic_loss = self._compute_ctc_loss(diacritic_logits, diacritic_labels, label_lengths)
        
        # Weighted combination
        weights = self.config.multitask_loss_weights
        total_loss = (weights[0] * final_loss + 
                    weights[1] * base_loss + 
                    weights[2] * diacritic_loss)
        
        # Log individual losses for monitoring
        if hasattr(self, '_log_individual_losses'):
            self._log_individual_losses(final_loss, base_loss, diacritic_loss, total_loss)
        
        return total_loss

    def _log_individual_losses(self, final_loss, base_loss, diacritic_loss, total_loss):
        """Log individual loss components (for debugging)"""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.debug(f"Loss breakdown - Final: {final_loss:.4f}, Base: {base_loss:.4f}, "
                        f"Diacritic: {diacritic_loss:.4f}, Total: {total_loss:.4f}")


    def save_pretrained(self, save_directory, **kwargs):
        """Enhanced save method that handles all hierarchical modes"""
        logger.info(f"Saving {self.__class__.__name__} model to: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        
        # Update config with current model state
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
            logger.warning("Vision encoder config not found, saving empty dict")
            self.config.vision_encoder_config = {}
        
        # Save model architecture info
        self.config.model_description = self.get_model_description()
        
        # Save the configuration
        self.config.save_pretrained(save_directory)
        
        # Create comprehensive state dict
        state_dict = self.state_dict()
        
        # Add metadata about which components are present
        metadata = {
            'hierarchical_mode': self.config.hierarchical_mode,
            'components_present': self._get_component_inventory(),
            'model_description': self.get_model_description(),
            'training_completed': True,
            'pytorch_version': torch.__version__,
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Save state dict with metadata
        full_checkpoint = {
            'model_state_dict': state_dict,
            'metadata': metadata,
            'config_dict': self.config.to_dict()
        }
        
        torch.save(full_checkpoint, os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save processor if available
        if hasattr(self, 'processor') and self.processor:
            try:
                self.processor.save_pretrained(save_directory)
                logger.info("Processor saved successfully")
            except Exception as e:
                logger.warning(f"Could not save processor: {e}")
        
        # Save vocabularies as separate files for easy access
        self._save_vocabularies(save_directory)
        
        # Save component-specific information
        self._save_component_info(save_directory)
        
        logger.info(f"Model saved successfully: {self.get_model_description()}")

    def _get_component_inventory(self):
        """Get inventory of which components are present in the model"""
        components = {
            'base_classifier': hasattr(self, 'base_classifier') and self.base_classifier is not None,
            'diacritic_classifier': hasattr(self, 'diacritic_classifier') and self.diacritic_classifier is not None,
            'final_classifier': hasattr(self, 'final_classifier') and self.final_classifier is not None,
            'dynamic_fusion': hasattr(self, 'dynamic_fusion') and self.dynamic_fusion is not None,
            'feature_enhancer': hasattr(self, 'feature_enhancer') and self.feature_enhancer is not None,
            'visual_diacritic_attention': hasattr(self, 'visual_diacritic_attention') and self.visual_diacritic_attention is not None,
            'character_diacritic_compatibility': hasattr(self, 'character_diacritic_compatibility') and self.character_diacritic_compatibility is not None,
            'few_shot_diacritic_adapter': hasattr(self, 'few_shot_diacritic_adapter') and self.few_shot_diacritic_adapter is not None,
            'diacritic_fusion': hasattr(self, 'diacritic_fusion') and self.diacritic_fusion is not None,
            'final_fusion': hasattr(self, 'final_fusion') and self.final_fusion is not None,
            'diacritic_condition_proj': hasattr(self, 'diacritic_condition_proj') and self.diacritic_condition_proj is not None,
            'diacritic_gate': hasattr(self, 'diacritic_gate') and self.diacritic_gate is not None,
        }
        return components

    def _save_vocabularies(self, save_directory):
        """Save vocabularies as separate JSON files"""
        vocabs_dir = os.path.join(save_directory, "vocabularies")
        os.makedirs(vocabs_dir, exist_ok=True)
        
        vocab_files = {
            'base_char_vocab.json': self.base_char_vocab,
            'diacritic_vocab.json': self.diacritic_vocab,
            'combined_char_vocab.json': self.combined_char_vocab
        }
        
        for filename, vocab in vocab_files.items():
            vocab_path = os.path.join(vocabs_dir, filename)
            try:
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {filename}")
            except Exception as e:
                logger.warning(f"Could not save {filename}: {e}")

    def _save_component_info(self, save_directory):
        """Save detailed component information"""
        info_path = os.path.join(save_directory, "model_info.json")
        
        component_info = {
            'hierarchical_mode': self.config.hierarchical_mode,
            'model_description': self.get_model_description(),
            'components_present': self._get_component_inventory(),
            'architecture_details': {
                'vision_encoder_layers': getattr(self.config, 'vision_encoder_layer_indices', []),
                'fusion_method': getattr(self.config, 'feature_fusion_method', 'unknown'),
                'transformer_layers': getattr(self.config, 'num_transformer_encoder_layers', 0),
                'conditioning_method': getattr(self.config, 'conditioning_method', 'none'),
            },
            'enhancement_modules': {
                'visual_diacritic_attention': self.config.use_visual_diacritic_attention,
                'character_diacritic_compatibility': self.config.use_character_diacritic_compatibility,
                'few_shot_diacritic_adapter': self.config.use_few_shot_diacritic_adapter,
                'dynamic_fusion': self.config.use_dynamic_fusion,
                'feature_enhancer': self.config.use_feature_enhancer,
            },
            'vocabulary_sizes': {
                'base_chars': len(self.base_char_vocab),
                'diacritics': len(self.diacritic_vocab), 
                'combined': len(self.combined_char_vocab)
            }
        }
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(component_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved model info to {info_path}")
        except Exception as e:
            logger.warning(f"Could not save model info: {e}")
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, strict_loading=True, **kwargs):
        """Enhanced loading method that handles all hierarchical modes"""
        logger.info(f"Loading {cls.__name__} from: {pretrained_model_name_or_path}")
        
        # Determine if this is a local directory or HuggingFace model
        is_local_dir = os.path.isdir(pretrained_model_name_or_path)
        
        if is_local_dir:
            # Local directory - direct file paths
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            checkpoint_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            model_info_path = os.path.join(pretrained_model_name_or_path, "model_info.json")
            
            # Check if this is our custom model or a base model
            is_our_custom_model = os.path.exists(model_info_path)
            
        else:
            # HuggingFace Hub model - need to download files
            try:
                from huggingface_hub import hf_hub_download
                
                # Always try to download config.json and pytorch_model.bin
                config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json")
                
                # For pytorch_model.bin, it might be named differently
                try:
                    checkpoint_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="pytorch_model.bin")
                except:
                    # Try alternative names
                    try:
                        checkpoint_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors")
                    except:
                        checkpoint_path = None
                        logger.warning("No model weights found - will use random initialization")
                
                # Try to download model_info.json but don't fail if it doesn't exist
                model_info_path = None
                try:
                    model_info_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path, 
                        filename="model_info.json",
                        local_files_only=False  # Allow download
                    )
                except:
                    # model_info.json doesn't exist - this is likely a base model
                    logger.info("No model_info.json found - treating as base model")
                    model_info_path = None
                
                is_our_custom_model = model_info_path is not None
                
            except Exception as e:
                logger.error(f"Error downloading from HuggingFace Hub: {e}")
                raise ValueError(f"Could not load model from {pretrained_model_name_or_path}: {e}")
        
        # Load model info if available
        model_info = {}
        if model_info_path and os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                logger.info(f"Loaded model info: {model_info.get('model_description', 'Unknown')}")
            except Exception as e:
                logger.warning(f"Could not load model info: {e}")
        
        # Load or create configuration
        if config is not None and isinstance(config, cls.config_class):
            loaded_config = config
            logger.info("Using provided config object")
            # Update with kwargs
            for key, value in kwargs.items():
                if hasattr(loaded_config, key):
                    setattr(loaded_config, key, value)
        elif config_path and os.path.exists(config_path):
            if is_our_custom_model:
                # This is our custom model - load our config
                try:
                    loaded_config = cls.config_class.from_pretrained(
                        os.path.dirname(config_path) if os.path.isfile(config_path) else config_path,
                        **kwargs
                    )
                    logger.info(f"Loaded custom config from: {config_path}")
                except Exception as e:
                    logger.warning(f"Could not load as custom config: {e}. Creating new config.")
                    loaded_config = cls._create_config_from_base_model(pretrained_model_name_or_path, **kwargs)
            else:
                # This is a base model (like TrOCR) - create our config
                logger.info("Base model detected - creating custom config")
                loaded_config = cls._create_config_from_base_model(pretrained_model_name_or_path, **kwargs)
        else:
            logger.warning(f"Config file not found. Creating new config.")
            # Ensure essential parameters are provided
            if 'combined_char_vocab' not in kwargs:
                raise ValueError("combined_char_vocab is required when config.json is not available")
            loaded_config = cls.config_class(**kwargs)
        
        # Handle vision_encoder_config conversion
        loaded_config = cls._process_vision_encoder_config(loaded_config)
        
        # Update vocab lists from kwargs if provided
        for vocab_name in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']:
            if vocab_name in kwargs and kwargs[vocab_name]:
                setattr(loaded_config, vocab_name, kwargs[vocab_name])
                setattr(loaded_config, f"{vocab_name}_size", len(kwargs[vocab_name]))
        
        # Load vocabularies from separate files if main config doesn't have them
        if not loaded_config.combined_char_vocab and is_local_dir:
            loaded_config = cls._load_vocabularies_from_files(loaded_config, pretrained_model_name_or_path)
        
        # Validate configuration compatibility
        if model_info:
            cls._validate_config_compatibility(loaded_config, model_info)
        
        # Instantiate model
        logger.info(f"Instantiating model with mode: {loaded_config.hierarchical_mode}")
        model = cls(loaded_config)
        
        # Load state dict if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = cls._load_model_weights(model, checkpoint_path, strict_loading, model_info, is_our_custom_model)
            
        else:
            logger.warning(f"No checkpoint found. Using base model initialization.")
        
        # Set vocabulary attributes
        model.base_char_vocab = loaded_config.base_char_vocab
        model.diacritic_vocab = loaded_config.diacritic_vocab
        model.combined_char_vocab = loaded_config.combined_char_vocab
        
        # Validate loaded model
        if model_info:
            cls._validate_loaded_model(model, model_info)
        
        logger.info(f"Successfully loaded model: {model.get_model_description()}")
        return model


    @classmethod
    def _process_vision_encoder_config(cls, config):
        """Process vision encoder config to ensure it's an object"""
        if hasattr(config, 'vision_encoder_config') and isinstance(config.vision_encoder_config, dict):
            vision_config_data = config.vision_encoder_config
            try:
                from transformers import AutoConfig
                
                # Special handling for TrOCR models
                if "trocr" in config.vision_encoder_name.lower():
                    from transformers import ViTConfig
                    config.vision_encoder_config = ViTConfig(**{k: v for k, v in vision_config_data.items() if k != "model_type"})
                    logger.info("Using ViTConfig for TrOCR-based model")
                else:
                    # Try to auto-detect config type
                    vision_model_type = vision_config_data.get("model_type", config.vision_encoder_name)
                    specific_config = AutoConfig.for_model(vision_model_type)
                    config.vision_encoder_config = specific_config(**vision_config_data)
                    logger.info(f"Using {specific_config.__class__.__name__} for vision encoder")
                    
            except Exception as e:
                logger.warning(f"Could not create specific vision config: {e}. Using generic config.")
                from transformers import PretrainedConfig
                base_config = PretrainedConfig()
                for k, v in vision_config_data.items():
                    setattr(base_config, k, v)
                config.vision_encoder_config = base_config
        
        return config

    @classmethod  
    def _load_vocabularies_from_files(cls, config, model_path):
        """Load vocabularies from separate JSON files"""
        vocabs_dir = os.path.join(model_path, "vocabularies") if os.path.isdir(model_path) else None
        
        if vocabs_dir and os.path.exists(vocabs_dir):
            vocab_files = {
                'base_char_vocab': 'base_char_vocab.json',
                'diacritic_vocab': 'diacritic_vocab.json', 
                'combined_char_vocab': 'combined_char_vocab.json'
            }
            
            for attr_name, filename in vocab_files.items():
                file_path = os.path.join(vocabs_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            vocab = json.load(f)
                        setattr(config, attr_name, vocab)
                        setattr(config, f"{attr_name}_size", len(vocab))
                        logger.info(f"Loaded {attr_name} from {filename} ({len(vocab)} items)")
                    except Exception as e:
                        logger.warning(f"Could not load {filename}: {e}")
        
        return config

    @classmethod
    def _validate_config_compatibility(cls, config, model_info):
        """Validate that config is compatible with saved model"""
        if not model_info:
            return
        
        saved_mode = model_info.get('hierarchical_mode')
        if saved_mode and saved_mode != config.hierarchical_mode:
            logger.warning(f"Config hierarchical_mode ({config.hierarchical_mode}) differs from saved model ({saved_mode})")
        
        saved_components = model_info.get('components_present', {})
        current_enhancements = {
            'visual_diacritic_attention': config.use_visual_diacritic_attention,
            'character_diacritic_compatibility': config.use_character_diacritic_compatibility,
            'few_shot_diacritic_adapter': config.use_few_shot_diacritic_adapter,
            'dynamic_fusion': config.use_dynamic_fusion,
            'feature_enhancer': config.use_feature_enhancer,
        }
        
        for component, expected in saved_components.items():
            if component in current_enhancements:
                current = current_enhancements[component]
                if current != expected:
                    logger.warning(f"Component {component}: config={current}, saved={expected}")

    @classmethod
    def _load_model_weights(cls, model, checkpoint_path, strict_loading, model_info, is_our_custom_model=None):
        """Load model weights with handling for both custom and base models"""
        logger.info(f"Loading model weights from: {checkpoint_path}")
        
        # Auto-detect if this is our custom model if not specified
        if is_our_custom_model is None:
            is_our_custom_model = model_info is not None and 'hierarchical_mode' in model_info
        
        try:
            # Handle different file formats
            if checkpoint_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path)
                    metadata = {}
                    logger.info("Loaded safetensors format")
                except ImportError:
                    logger.error("safetensors not available but .safetensors file provided")
                    raise
            else:
                # checkpoint = torch.load(checkpoint_path, map_location="cpu")
                try:
                    # Try with weights_only=True first (secure)
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only=True: {e}")
                    logger.info("Falling back to weights_only=False (less secure but compatible)")
                    # Fall back to weights_only=False for compatibility
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Our custom format
                    state_dict = checkpoint['model_state_dict']
                    metadata = checkpoint.get('metadata', {})
                    logger.info(f"Loaded custom checkpoint with metadata: {metadata.get('model_description', 'Unknown')}")
                else:
                    # Standard format (like TrOCR)
                    state_dict = checkpoint
                    metadata = {}
                    logger.info("Loaded standard checkpoint format")
            
            if is_our_custom_model:
                # Loading our custom model - expect exact match
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_loading)
            else:
                # Loading from base model - only load vision encoder weights
                vision_encoder_state = {}
                for key, value in state_dict.items():
                    if key.startswith('encoder.') or key.startswith('vision_model.'):
                        # Map base model keys to our vision_encoder
                        new_key = key.replace('encoder.', 'vision_encoder.').replace('vision_model.', 'vision_encoder.')
                        vision_encoder_state[new_key] = value
                    elif key.startswith('vision_encoder.'):
                        vision_encoder_state[key] = value
                
                if vision_encoder_state:
                    missing_keys, unexpected_keys = model.load_state_dict(vision_encoder_state, strict=False)
                    logger.info(f"Loaded {len(vision_encoder_state)} vision encoder weights from base model")
                else:
                    logger.warning("No compatible vision encoder weights found in base model")
                    missing_keys, unexpected_keys = [], []
            
            # Report loading results
            if missing_keys:
                if is_our_custom_model:
                    logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                    logger.debug(f"Missing keys: {missing_keys[:10]}...")
                else:
                    logger.info(f"Missing keys (expected for base model): {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected keys: {unexpected_keys[:10]}...")
            
            if is_our_custom_model and not missing_keys and not unexpected_keys:
                logger.info("All model weights loaded successfully")
            elif not is_our_custom_model:
                logger.info("Vision encoder weights loaded from base model, other components randomly initialized")
            elif not strict_loading:
                logger.info("Model weights loaded with warnings (strict=False)")
            else:
                logger.error("Model weight loading failed with strict=True")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            if strict_loading:
                raise
            logger.warning("Continuing with random initialization due to loading error")
        
        return model

    @classmethod
    def _validate_loaded_model(cls, model, model_info):
        """Validate that the loaded model is working correctly"""
        try:
            # Basic validation: check that model can do a forward pass
            dummy_input = torch.randn(1, 3, 384, 384)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            # Check output structure
            required_keys = ['logits', 'base_logits', 'diacritic_logits']
            for key in required_keys:
                if key not in outputs:
                    logger.warning(f"Missing output key: {key}")
            
            logger.info("Model validation passed")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            logger.warning("Model may not function correctly")

    def get_model_description(self):
        """Get a descriptive name based on current configuration"""
        mode = self.config.hierarchical_mode
        enhancements = []
        
        if getattr(self.config, 'use_dynamic_fusion', False):
            enhancements.append("DynFusion")
        if getattr(self.config, 'use_feature_enhancer', False):
            enhancements.append("FeatEnh")
        if getattr(self.config, 'use_visual_diacritic_attention', False):
            enhancements.append("VDA")
        if getattr(self.config, 'use_character_diacritic_compatibility', False):
            enhancements.append("CDC")
        if getattr(self.config, 'use_few_shot_diacritic_adapter', False):
            enhancements.append("FSA")
        
        enhancement_str = "+".join(enhancements) if enhancements else "Base"
        
        mode_names = {
            "enhanced_single": "Enhanced-CTC",
            "parallel": "Parallel-CTC",
            "sequential": "Sequential-HCTC", 
            "multitask": "MultiTask-CTC"
        }
        
        base_name = mode_names.get(mode, f"{mode}-CTC")
        return f"{base_name}-{enhancement_str}"

    @classmethod
    def _create_config_from_base_model(cls, base_model_name, **kwargs):
        """Create our config when loading from a base model like TrOCR"""
        
        # Set defaults for base model loading
        config_kwargs = {
            'vision_encoder_name': base_model_name,
            'hierarchical_mode': kwargs.get('hierarchical_mode', 'enhanced_single'),
            'use_dynamic_fusion': kwargs.get('use_dynamic_fusion', False),
            'use_feature_enhancer': kwargs.get('use_feature_enhancer', False),
            'use_visual_diacritic_attention': kwargs.get('use_visual_diacritic_attention', False),
            'use_character_diacritic_compatibility': kwargs.get('use_character_diacritic_compatibility', False),
            'use_few_shot_diacritic_adapter': kwargs.get('use_few_shot_diacritic_adapter', False),
            # Add other required parameters
            **kwargs
        }
        
        # Ensure required vocabularies are present
        if 'combined_char_vocab' not in config_kwargs:
            raise ValueError("combined_char_vocab must be provided when loading from base model")
        
        logger.info(f"Creating config for base model: {base_model_name}")
        return cls.config_class(**config_kwargs)

# --- Combined Multi-Scale Model Class ---
class HierarchicalCtcMultiScaleOcrModel(HierarchicalCtcTransformerOcrModel):
    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        logger.info(f"Initialized {self.__class__.__name__} (inherits Grad-CAM from parent).")