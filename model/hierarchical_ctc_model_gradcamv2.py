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
from .diacritic_attention import ( # Make sure these paths are correct for your project
    VisualDiacriticAttention,
    CharacterDiacriticCompatibility,
    FewShotDiacriticAdapter
)
from .dynamic_fusion import ( # Make sure these paths are correct for your project
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
        self.vision_encoder_config = vision_encoder_config 
        self.use_visual_diacritic_attention = use_visual_diacritic_attention
        self.use_character_diacritic_compatibility = use_character_diacritic_compatibility
        self.use_few_shot_diacritic_adapter = use_few_shot_diacritic_adapter
        self.num_few_shot_prototypes = num_few_shot_prototypes
        if not self.combined_char_vocab:
            logger.warning("Combined character vocabulary is empty during config init.")

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

        try:
            self.processor = AutoProcessor.from_pretrained(config.vision_encoder_name, trust_remote_code=True)
            base_v_model = VisionEncoderDecoderModel.from_pretrained(config.vision_encoder_name, trust_remote_code=True)
            self.vision_encoder = base_v_model.encoder
            if isinstance(config.vision_encoder_config, dict):
                vision_config_class = self.vision_encoder.config.__class__ 
                self.config.vision_encoder_config = vision_config_class(**config.vision_encoder_config)
            elif self.config.vision_encoder_config is None:
                 self.config.vision_encoder_config = self.vision_encoder.config
            logger.info("Processor and Vision Encoder loaded.")
            del base_v_model
        except Exception as e:
            logger.error(f"Failed loading base model components: {e}", exc_info=True); raise

        encoder_hidden_size = self.config.vision_encoder_config.hidden_size
        num_fusion_layers = len(config.vision_encoder_layer_indices)
        self.dynamic_fusion = None; self.fusion_projection = None; self.pre_fusion_projections = None; self.fusion_bilinear = None
        sequence_model_input_size = config.transformer_d_model

        if config.use_dynamic_fusion and num_fusion_layers > 1:
            logger.info(f"Using Dynamic Multi-Scale Fusion ({num_fusion_layers} layers -> {config.transformer_d_model} dim)")
            self.dynamic_fusion = DynamicMultiScaleFusion(encoder_hidden_size, num_fusion_layers, config.transformer_d_model)
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
        else: 
             if encoder_hidden_size != config.transformer_d_model:
                  self.fusion_projection = nn.Linear(encoder_hidden_size, config.transformer_d_model)
                  logger.info(f"Projecting single/last selected encoder layer ({encoder_hidden_size}) to {config.transformer_d_model}")
             else: sequence_model_input_size = encoder_hidden_size 

        self.feature_enhancer = None
        if config.use_feature_enhancer:
            logger.info("Using Local Feature Enhancer")
            self.feature_enhancer = LocalFeatureEnhancer(sequence_model_input_size, config.diacritic_vocab_size if config.diacritic_vocab_size > 0 else 1) # Handle empty diacritic vocab for enhancer init

        if config.positional_encoding_type == "sinusoidal_1d": self.pos_encoder = SinusoidalPositionalEncoding1D(sequence_model_input_size, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        elif config.positional_encoding_type == "learned_1d": self.pos_encoder = LearnedPositionalEncoding1D(sequence_model_input_size, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        else: self.pos_encoder = nn.Identity()
        logger.info(f"Positional Encoding: {config.positional_encoding_type}")

        logger.info(f"Adding {config.num_transformer_encoder_layers} Transformer Encoder layers (Dim: {sequence_model_input_size})")
        if sequence_model_input_size > 0 and sequence_model_input_size % config.transformer_nhead != 0 : # Added check for > 0
            raise ValueError(f"Transformer d_model ({sequence_model_input_size}) must be divisible by nhead ({config.transformer_nhead}) and positive.")
        elif sequence_model_input_size <=0:
             logger.warning(f"Sequence model input size is {sequence_model_input_size}. Transformer encoder will be nn.Identity()")
             self.transformer_encoder = nn.Identity()
        else:
            tf_encoder_layer = TransformerEncoderLayer(sequence_model_input_size, config.transformer_nhead, config.transformer_dim_feedforward, config.transformer_dropout, F.gelu, batch_first=True)
            self.transformer_encoder = TransformerEncoder(tf_encoder_layer, config.num_transformer_encoder_layers, nn.LayerNorm(sequence_model_input_size))
        transformer_output_size = sequence_model_input_size

        shared_layers_modules = []
        current_shared_size = transformer_output_size
        if current_shared_size > 0: # Only add shared layers if input is valid
            for i in range(config.num_shared_layers):
                out_s = config.shared_hidden_size
                shared_layers_modules.extend([nn.Linear(current_shared_size, out_s), nn.LayerNorm(out_s), nn.GELU(), nn.Dropout(config.classifier_dropout)])
                current_shared_size = out_s
        self.shared_layer = nn.Sequential(*shared_layers_modules) if config.num_shared_layers > 0 and current_shared_size > 0 else nn.Identity()
        shared_output_size = current_shared_size
        logger.info(f"Shared Layer(s): {config.num_shared_layers}, Output Dim: {shared_output_size}")

        self.base_classifier = nn.Linear(shared_output_size, config.base_char_vocab_size) if config.base_char_vocab_size > 0 and shared_output_size > 0 else nn.Identity()
        logger.info(f"Base Classifier (In: {shared_output_size}, Out: {config.base_char_vocab_size})")

        self.diacritic_gate = None; self.diacritic_condition_proj = None
        diacritic_head_input_size = shared_output_size
        if config.conditioning_method == 'concat_proj' and shared_output_size > 0 and config.base_char_vocab_size > 0:
            self.diacritic_condition_proj = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.LayerNorm(shared_output_size), nn.GELU(), nn.Dropout(config.classifier_dropout))
            logger.info(f"'concat_proj' conditioning for diacritic head -> {shared_output_size} dim.")
        elif config.conditioning_method == 'gate' and shared_output_size > 0 and config.base_char_vocab_size > 0:
            self.diacritic_gate = nn.Sequential(nn.Linear(shared_output_size + config.base_char_vocab_size, shared_output_size), nn.Sigmoid())
            logger.info(f"'gate' conditioning for diacritic head -> {shared_output_size} dim.")
        else: logger.info(f"'none' conditioning for diacritic head -> {shared_output_size} dim.")

        self.visual_diacritic_attention = None; self.character_diacritic_compatibility = None; self.few_shot_diacritic_adapter = None
        if config.use_visual_diacritic_attention and diacritic_head_input_size > 0 and config.diacritic_vocab_size > 0: self.visual_diacritic_attention = VisualDiacriticAttention(diacritic_head_input_size, config.diacritic_vocab_size); logger.info("Visual Diacritic Attn: ON")
        if config.use_character_diacritic_compatibility and config.base_char_vocab_size > 0 and config.diacritic_vocab_size > 0 and diacritic_head_input_size > 0: self.character_diacritic_compatibility = CharacterDiacriticCompatibility(config.base_char_vocab_size, config.diacritic_vocab_size, diacritic_head_input_size, self.base_char_vocab, self.diacritic_vocab); logger.info("Char-Diac Compat: ON")
        if config.use_few_shot_diacritic_adapter and diacritic_head_input_size > 0 and config.diacritic_vocab_size > 0: self.few_shot_diacritic_adapter = FewShotDiacriticAdapter(diacritic_head_input_size, config.diacritic_vocab_size, config.num_few_shot_prototypes); logger.info("Few-Shot Adapter: ON")

        self.diacritic_classifier = nn.Linear(diacritic_head_input_size, config.diacritic_vocab_size) if config.diacritic_vocab_size > 0 and diacritic_head_input_size > 0 else nn.Identity()
        logger.info(f"Diacritic Classifier (In: {diacritic_head_input_size}, Out: {config.diacritic_vocab_size})")

        self.final_classifier = nn.Linear(shared_output_size, config.combined_char_vocab_size) if config.combined_char_vocab_size > 0 and shared_output_size > 0 else nn.Identity()
        logger.info(f"Final Combined Classifier (In: {shared_output_size}, Out: {config.combined_char_vocab_size})")

        self.activation_hook_handles = []
        self.activations = None
        self._init_weights()

    def _init_weights(self): 
        logger.debug("Initializing weights for custom layers...")
        # Add specific initializations if needed

    def _save_activation(self, module, input, output):
        self.activations = output
        if self.activations is not None and self.activations.requires_grad:
            self.activations.retain_grad()

    def _register_hooks(self, target_layer_module):
        self.clear_hooks()
        logger.debug(f"Registering forward hook for Grad-CAM on: {target_layer_module.__class__.__name__}")
        handle_f = target_layer_module.register_forward_hook(self._save_activation)
        self.activation_hook_handles.append(handle_f)

    def clear_hooks(self):
        for handle in self.activation_hook_handles: handle.remove()
        self.activation_hook_handles = []
        self.activations = None 

    def get_activations_and_gradients(self):
        current_gradients = None
        current_activations = None
        if self.activations is not None:
            current_activations = self.activations.detach().clone() 
            if self.activations.grad is not None:
                current_gradients = self.activations.grad.detach().clone()
                logger.debug("Successfully retrieved .grad from hooked activations.")
            else:
                logger.warning("Gradients (.grad) for hooked activations were not populated.")
        else:
            logger.warning("Activations were None when trying to get gradients for Grad-CAM.")
        return current_activations, current_gradients

    def forward(self, pixel_values, labels=None, label_lengths=None,
                return_diacritic_attention=False,
                grad_cam_target_layer_module_name=None,
                ### ADDED ARGUMENT ###
                grad_cam_logit_target_type="diacritic" # "final", "diacritic", "base"
                ):

        if grad_cam_target_layer_module_name and not pixel_values.requires_grad:
            pixel_values.requires_grad_(True) 

        encoder_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        all_hidden_states = encoder_outputs.hidden_states

        num_enc_layers = len(all_hidden_states)
        valid_indices = sorted(list(set(idx if idx >= 0 else num_enc_layers + idx for idx in self.config.vision_encoder_layer_indices)))
        valid_indices = [i for i in valid_indices if 0 <= i < num_enc_layers] 
        if not valid_indices: features_to_fuse = [all_hidden_states[-1]]
        else: features_to_fuse = [all_hidden_states[i] for i in valid_indices]

        target_module_for_grad_cam = None
        if grad_cam_target_layer_module_name:
            if hasattr(self, grad_cam_target_layer_module_name):
                target_module_for_grad_cam = getattr(self, grad_cam_target_layer_module_name)
            elif '.' in grad_cam_target_layer_module_name:
                try:
                    current_obj = self
                    for part in grad_cam_target_layer_module_name.split('.'):
                        if part.isdigit(): current_obj = current_obj[int(part)] 
                        else: current_obj = getattr(current_obj, part)
                    target_module_for_grad_cam = current_obj
                except (AttributeError, IndexError, ValueError) as e:
                    logger.warning(f"Could not resolve nested Grad-CAM target '{grad_cam_target_layer_module_name}': {e}")
            
            if target_module_for_grad_cam is not None and isinstance(target_module_for_grad_cam, nn.Module):
                self._register_hooks(target_module_for_grad_cam)
            elif target_module_for_grad_cam is None:
                 logger.warning(f"Grad-CAM target layer '{grad_cam_target_layer_module_name}' not found or is not an nn.Module.")

        fused_features = None
        if self.dynamic_fusion: fused_features = self.dynamic_fusion(features_to_fuse)
        elif self.fusion_projection and self.config.feature_fusion_method == "concat_proj":
            fused_features = self.fusion_projection(torch.cat(features_to_fuse, dim=-1))
        elif self.pre_fusion_projections and self.config.feature_fusion_method == "add":
            projected_features = [proj(feat) for proj, feat in zip(self.pre_fusion_projections, features_to_fuse)]
            if projected_features: fused_features = torch.stack(projected_features).mean(0)
            else: fused_features = features_to_fuse[-1] # Fallback if no projections
        elif self.fusion_bilinear and len(features_to_fuse) >= 2: fused_features = self.fusion_bilinear(features_to_fuse[0], features_to_fuse[1])
        elif self.fusion_projection: fused_features = self.fusion_projection(features_to_fuse[-1]) 
        else: fused_features = features_to_fuse[-1] 

        enhanced_features = self.feature_enhancer(fused_features) if self.feature_enhancer and not isinstance(self.feature_enhancer, nn.Identity) else fused_features
        features_with_pos = self.pos_encoder(enhanced_features) if not isinstance(self.pos_encoder, nn.Identity) else enhanced_features
        
        if isinstance(self.transformer_encoder, nn.Identity):
            transformer_output = features_with_pos
        else:
            transformer_output = self.transformer_encoder(features_with_pos)
        
        shared_features = self.shared_layer(transformer_output) if not isinstance(self.shared_layer, nn.Identity) else transformer_output

        base_logits = self.base_classifier(shared_features) if not isinstance(self.base_classifier, nn.Identity) else torch.empty(0) # Handle Identity
        
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat_proj' and self.diacritic_condition_proj and not isinstance(self.base_classifier, nn.Identity):
            diacritic_input_features = self.diacritic_condition_proj(torch.cat((shared_features, F.softmax(base_logits, dim=-1)), dim=-1))
        elif self.config.conditioning_method == 'gate' and self.diacritic_gate and not isinstance(self.base_classifier, nn.Identity):
            diacritic_input_features = shared_features * self.diacritic_gate(torch.cat((shared_features, F.softmax(base_logits, dim=-1)), dim=-1))

        standard_diacritic_logits = self.diacritic_classifier(diacritic_input_features) if not isinstance(self.diacritic_classifier, nn.Identity) else torch.empty(0)
        current_diacritic_logits = standard_diacritic_logits
        vda_raw_output, visual_diacritic_attention_maps = None, None

        if self.visual_diacritic_attention:
            res = self.visual_diacritic_attention(diacritic_input_features, return_attention_weights=True)
            vda_raw_output, visual_diacritic_attention_maps = res if isinstance(res, tuple) else (res, None)
            if vda_raw_output is not None and current_diacritic_logits.numel() > 0:
                 current_diacritic_logits = current_diacritic_logits + vda_raw_output
            elif vda_raw_output is not None:
                 current_diacritic_logits = vda_raw_output


        if self.character_diacritic_compatibility and not isinstance(self.base_classifier, nn.Identity) and current_diacritic_logits.numel() > 0:
             current_diacritic_logits = current_diacritic_logits + self.character_diacritic_compatibility(base_logits, diacritic_input_features)
        if self.few_shot_diacritic_adapter and current_diacritic_logits.numel() > 0 :
             current_diacritic_logits = current_diacritic_logits + self.few_shot_diacritic_adapter(diacritic_input_features)
        diacritic_logits = current_diacritic_logits

        final_logits = self.final_classifier(shared_features) if not isinstance(self.final_classifier, nn.Identity) else torch.empty(0)

        loss = None
        if labels is not None and label_lengths is not None and final_logits.numel() > 0:
            log_probs = final_logits.log_softmax(dim=2).permute(1, 0, 2)
            device = log_probs.device; time_steps = log_probs.size(0); bs = log_probs.size(1)
            input_lengths = torch.full((bs,), time_steps, dtype=torch.long, device=device)
            labels_dev, label_lengths_dev = labels.to(device), label_lengths.to(device)
            input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
            label_lengths_clamped = torch.clamp(label_lengths_dev, max=labels_dev.size(1))
            try:
                ctc_loss_fn = nn.CTCLoss(blank=self.config.blank_idx, reduction='mean', zero_infinity=True)
                valid_mask = label_lengths_dev > 0
                if torch.any(valid_mask):
                    loss = ctc_loss_fn(log_probs[:, valid_mask, :], labels_dev[valid_mask], input_lengths_clamped[valid_mask], label_lengths_clamped[valid_mask])
                    if torch.isnan(loss) or torch.isinf(loss): loss = torch.tensor(0.0, device=device, requires_grad=True)
                else: loss = torch.tensor(0.0, device=device, requires_grad=True)
            except Exception as e: logger.error(f"CTC loss error: {e}"); loss = torch.tensor(0.0, device=device, requires_grad=True)

        output_dict = {'loss': loss, 'logits': final_logits, 'base_logits': base_logits, 'diacritic_logits': diacritic_logits}
        if return_diacritic_attention and visual_diacritic_attention_maps is not None:
            output_dict['visual_diacritic_attention_maps'] = visual_diacritic_attention_maps
        
        ### MODIFIED: Select logits for Grad-CAM based on grad_cam_logit_target_type ###
        if grad_cam_target_layer_module_name:
            if grad_cam_logit_target_type == "final" and final_logits.numel() > 0:
                output_dict['grad_cam_target_logits'] = final_logits
            elif grad_cam_logit_target_type == "base" and base_logits.numel() > 0:
                output_dict['grad_cam_target_logits'] = base_logits
            elif grad_cam_logit_target_type == "diacritic" and diacritic_logits.numel() > 0:
                 # Prefer VDA output if available and targeted, otherwise full diacritic logits
                if self.visual_diacritic_attention and vda_raw_output is not None:
                     output_dict['grad_cam_target_logits'] = vda_raw_output
                else:
                     output_dict['grad_cam_target_logits'] = diacritic_logits
            elif final_logits.numel() > 0: # Fallback to final_logits if specified target is empty
                logger.warning(f"Grad-CAM target logits '{grad_cam_logit_target_type}' are empty. Falling back to 'final_logits'.")
                output_dict['grad_cam_target_logits'] = final_logits
            else:
                logger.warning(f"No valid logits found for Grad-CAM target type '{grad_cam_logit_target_type}' and final_logits also empty.")
                output_dict['grad_cam_target_logits'] = None
        return output_dict

    def save_pretrained(self, save_directory, **kwargs):
        logger.info(f"Saving {self.__class__.__name__} model to: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        self.config.base_char_vocab = self.base_char_vocab
        self.config.diacritic_vocab = self.diacritic_vocab
        self.config.combined_char_vocab = self.combined_char_vocab
        self.config.base_char_vocab_size = len(self.base_char_vocab)
        self.config.diacritic_vocab_size = len(self.diacritic_vocab)
        self.config.combined_char_vocab_size = len(self.combined_char_vocab)
        if hasattr(self.vision_encoder, 'config') and self.vision_encoder.config is not None:
            self.config.vision_encoder_config = self.vision_encoder.config.to_dict()
        else:
            logger.warning("Vision encoder config not found or is None, cannot save to dict.")
            self.config.vision_encoder_config = {} 

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
        loaded_config_dict = {} 

        if config is not None and isinstance(config, cls.config_class):
            loaded_config = config
            logger.info("Using provided config object.")
            for key, value in kwargs.items():
                if hasattr(loaded_config, key): setattr(loaded_config, key, value)
                else: loaded_config_dict[key] = value 
        elif os.path.exists(config_path):
            loaded_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
            logger.info(f"Loaded config from file, applied/overrode with kwargs: {kwargs}")
        else: 
            logger.warning(f"Config file not found at {config_path}. Initializing new config.")
            loaded_config_dict.update(kwargs)
            if 'combined_char_vocab' not in loaded_config_dict:
                raise ValueError("combined_char_vocab is required if config.json is not present.")
            loaded_config = cls.config_class(**loaded_config_dict)

        if hasattr(loaded_config, 'vision_encoder_config') and isinstance(loaded_config.vision_encoder_config, dict):
            vision_config_data = loaded_config.vision_encoder_config
            try: 
                from transformers import AutoConfig
                # Determine vision_model_type from vision_encoder_name if not in config dict
                vision_model_type_from_name = loaded_config.vision_encoder_name.split('/')[-1].split('-')[0]
                vision_model_type = vision_config_data.get("model_type", vision_model_type_from_name)
                
                if "trocr" in loaded_config.vision_encoder_name.lower() and "model_type" not in vision_config_data:
                    # TrOCR uses ViT, ensure ViTConfig is used if model_type is not explicit
                    from transformers import ViTConfig
                    logger.info(f"TrOCR model detected ('{loaded_config.vision_encoder_name}'), ensuring ViTConfig for vision encoder.")
                    vision_config_obj = ViTConfig(**{k:v for k,v in vision_config_data.items() if k != "model_type"}) # Avoid passing model_type if it's wrong
                else:
                    vision_config_class = AutoConfig.for_model(vision_model_type)._model_config_class
                    vision_config_obj = vision_config_class(**vision_config_data)

                loaded_config.vision_encoder_config = vision_config_obj
                logger.info(f"Converted vision_encoder_config dict to {vision_config_obj.__class__.__name__} object.")
            except Exception as e_conf:
                logger.warning(f"Could not auto-detect specific vision config type (model_type: {vision_model_type if 'vision_model_type' in locals() else 'N/A'}): {e_conf}. Using PretrainedConfig as base.")
                base_vision_conf = PretrainedConfig()
                for k_vc, v_vc in vision_config_data.items(): setattr(base_vision_conf, k_vc, v_vc)
                loaded_config.vision_encoder_config = base_vision_conf
        
        for vocab_name in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']:
            if vocab_name in kwargs and kwargs[vocab_name]:
                setattr(loaded_config, vocab_name, kwargs[vocab_name])
                setattr(loaded_config, f"{vocab_name}_size", len(kwargs[vocab_name]))

        model = cls(loaded_config) 

        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path) and os.path.exists(config_path): 
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

class HierarchicalCtcMultiScaleOcrModel(HierarchicalCtcTransformerOcrModel):
    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        logger.info(f"Initialized {self.__class__.__name__} (inherits Grad-CAM from parent).")