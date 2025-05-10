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
# --- NEW: Import Transformer specific layers ---
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os
import json
import logging
import inspect # For checking kwargs in from_pretrained fallback
import math

logger = logging.getLogger(__name__)

# --- Configuration Class (Updated for Transformer Encoder) ---
class HierarchicalCtcOcrConfig(PretrainedConfig):
    model_type = "hierarchical_ctc_transformer" # Reflects Transformer change

    def __init__(
        self,
        vision_encoder_name='microsoft/trocr-base-handwritten',
        base_char_vocab=None,
        diacritic_vocab=None,
        combined_char_vocab=None,
        # Vision Feature Fusion
        vision_encoder_layer_indices=[-1, -4],
        feature_fusion_method="concat_proj",
        # --- REMOVED RNN Params ---
        # --- NEW: Transformer Encoder Params ---
        num_transformer_encoder_layers=4,
        transformer_d_model=512, # Dimension for transformer (input after fusion/projection)
        transformer_nhead=8,
        transformer_dim_feedforward=2048,
        transformer_dropout=0.1,
        positional_encoding_type="sinusoidal_1d",
        max_seq_len_for_pos_enc=768, # Max sequence length for pos encoding (e.g., num_patches from ViT)
        # --- End New Transformer Params ---
        shared_hidden_size=512, # Output of Shared Layer
        num_shared_layers=1,
        conditioning_method="concat_proj",
        classifier_dropout=0.1,
        blank_idx=0,
        vision_encoder_config=None, # Will be populated
        # Diacritic Enhancement Flags
        use_visual_diacritic_attention=False,
        use_character_diacritic_compatibility=False,
        use_few_shot_diacritic_adapter=False,
        num_few_shot_prototypes=10,
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

# --- Positional Encoding Classes (Copied from previous suggestion) ---
class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Adjusted for batch_first=True: pe shape [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: Tensor, shape [batch_size, seq_len, embedding_dim] """
        x = x + self.pe[:, :x.size(1), :] # Broadcasting pe to batch dim
        return self.dropout(x)

class LearnedPositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        # No need to register positions as buffer if using it directly
        # self.register_buffer('positions', torch.arange(max_len))

    def forward(self, x):
        """ x: Tensor, shape [batch_size, seq_len, embedding_dim] """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # [1, seq_len]
        x = x + self.pos_embedding(positions) # pos_embedding(positions) will be [1, seq_len, d_model]
        return self.dropout(x)

# Placeholder for potential advanced diacritic components (from your existing code)
class VisualDiacriticAttention(nn.Module): # Placeholder
    def __init__(self, feature_dim, diacritic_vocab_size): super().__init__(); self.fc = nn.Linear(feature_dim, diacritic_vocab_size)
    def forward(self, x): return self.fc(x)
class CharacterDiacriticCompatibility(nn.Module): # Placeholder
    def __init__(self, base_vocab_size, diacritic_vocab_size, shared_dim, base_char_vocab, diacritic_vocab): super().__init__(); self.fc = nn.Linear(base_vocab_size, diacritic_vocab_size) # Dummy
    def forward(self, base_logits, shared_features): return self.fc(base_logits) # Dummy
class FewShotDiacriticAdapter(nn.Module): # Placeholder
    def __init__(self, feature_dim, diacritic_vocab_size, num_prototypes): super().__init__(); self.fc = nn.Linear(feature_dim, diacritic_vocab_size)
    def forward(self, x): return self.fc(x)


# --- Model Class (Hierarchical + Multi-Scale + Transformer Encoder) ---
class HierarchicalCtcTransformerOcrModel(PreTrainedModel):
    config_class = HierarchicalCtcOcrConfig

    def __init__(self, config: HierarchicalCtcOcrConfig):
        super().__init__(config)
        self.config = config
        if not config.combined_char_vocab: raise ValueError("Combined vocab missing.")
        if config.conditioning_method != 'none' and (not config.base_char_vocab or not config.diacritic_vocab): raise ValueError("Base/Diac vocabs missing for conditioning.")
        self.base_char_vocab = config.base_char_vocab; self.diacritic_vocab = config.diacritic_vocab; self.combined_char_vocab = config.combined_char_vocab

        logger.info(f"Initializing {self.__class__.__name__}...")

        # --- Load Base Components ---
        try:
            self.processor = AutoProcessor.from_pretrained(config.vision_encoder_name)
            base_v_model = VisionEncoderDecoderModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder = base_v_model.encoder
            if self.config.vision_encoder_config is None: self.config.vision_encoder_config = self.vision_encoder.config
            logger.info("Processor and Vision Encoder loaded.")
            del base_v_model
        except Exception as e: logger.error(f"Failed loading base: {e}", exc_info=True); raise

        # --- Feature Fusion Layer ---
        encoder_hidden_size = self.config.vision_encoder_config.hidden_size
        num_fusion_layers = len(config.vision_encoder_layer_indices)
        fusion_input_size_raw = encoder_hidden_size * num_fusion_layers
        self.fusion_projection = None; self.pre_fusion_projections = None; self.fusion_bilinear = None

        # Determine input size for positional encoding / transformer
        sequence_model_input_size = config.transformer_d_model

        if config.feature_fusion_method == "concat_proj" and num_fusion_layers > 1:
            self.fusion_projection = nn.Linear(fusion_input_size_raw, config.transformer_d_model)
            logger.info(f"Using 'concat_proj' fusion (Raw In: {fusion_input_size_raw}, Projected Out: {config.transformer_d_model})")
        elif config.feature_fusion_method == "add" and num_fusion_layers > 1:
            if encoder_hidden_size != config.transformer_d_model:
                self.pre_fusion_projections = nn.ModuleList([nn.Linear(encoder_hidden_size, config.transformer_d_model) for _ in range(num_fusion_layers)])
            logger.info("Using 'add' feature fusion (Out: {config.transformer_d_model})")
        elif config.feature_fusion_method == "bilinear" and num_fusion_layers == 2:
             self.fusion_bilinear = nn.Bilinear(encoder_hidden_size, encoder_hidden_size, config.transformer_d_model)
             logger.info("Using 'bilinear' feature fusion (Out: {config.transformer_d_model})")
        else: # 'none' or single layer from fusion_indices
            if config.feature_fusion_method != 'none' and num_fusion_layers > 1: logger.warning(f"Fusion method '{config.feature_fusion_method}' not fully handled or invalid. Using last selected layer.")
            if encoder_hidden_size != config.transformer_d_model:
                 self.fusion_projection = nn.Linear(encoder_hidden_size, config.transformer_d_model)
                 logger.info(f"Projecting last selected encoder layer to transformer_d_model: {config.transformer_d_model}")
            # If no projection needed, sequence_model_input_size is encoder_hidden_size, which must match transformer_d_model

        # --- Positional Encoding ---
        if config.positional_encoding_type == "sinusoidal_1d":
            self.pos_encoder = SinusoidalPositionalEncoding1D(sequence_model_input_size, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        elif config.positional_encoding_type == "learned_1d":
            self.pos_encoder = LearnedPositionalEncoding1D(sequence_model_input_size, config.transformer_dropout, config.max_seq_len_for_pos_enc)
        else:
            logger.warning(f"Unknown positional_encoding_type: {config.positional_encoding_type}. Applying nn.Identity().")
            self.pos_encoder = nn.Identity()
        logger.info(f"Using Positional Encoding: {config.positional_encoding_type}")

        # --- Transformer Encoder Layers ---
        logger.info(f"Adding {config.num_transformer_encoder_layers} Transformer Encoder layers...")
        if sequence_model_input_size % config.transformer_nhead != 0:
            raise ValueError(f"Transformer d_model ({sequence_model_input_size}) must be divisible by nhead ({config.transformer_nhead})")
        encoder_layer_norm = nn.LayerNorm(sequence_model_input_size) # Optional: Norm before/after encoder stack
        encoder_layer = TransformerEncoderLayer(
            d_model=sequence_model_input_size, nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward, dropout=config.transformer_dropout,
            activation=F.gelu, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.num_transformer_encoder_layers, norm=encoder_layer_norm)
        transformer_output_size = sequence_model_input_size

        # --- Shared Feature Layer (Input from Transformer Encoder) ---
        shared_layers_modules = []
        current_shared_size = transformer_output_size
        for i in range(config.num_shared_layers):
            shared_layers_modules.extend([
                nn.Linear(current_shared_size, config.shared_hidden_size),
                nn.LayerNorm(config.shared_hidden_size), nn.GELU(),
                nn.Dropout(config.classifier_dropout)
            ])
            current_shared_size = config.shared_hidden_size
        self.shared_layer = nn.Sequential(*shared_layers_modules)
        logger.info(f"Added {config.num_shared_layers} shared layer(s) (Output: {config.shared_hidden_size})")

        # --- Hierarchical Heads & Final Classifier (Same logic as before) ---
        self.base_classifier = nn.Linear(config.shared_hidden_size, config.base_char_vocab_size)
        logger.info(f"Base Classifier Head (Out: {config.base_char_vocab_size})")

        conditioning_input_size = config.shared_hidden_size
        self.diacritic_gate = None; self.diacritic_condition_proj = None
        if config.conditioning_method == 'concat_proj':
            concat_size = config.shared_hidden_size + config.base_char_vocab_size
            self.diacritic_condition_proj = nn.Sequential(nn.Linear(concat_size, config.shared_hidden_size), nn.LayerNorm(config.shared_hidden_size), nn.GELU(), nn.Dropout(config.classifier_dropout))
            diacritic_head_input_size = config.shared_hidden_size
            logger.info("'concat_proj' conditioning for diacritic head.")
        elif config.conditioning_method == 'gate':
            self.diacritic_gate = nn.Sequential(nn.Linear(config.shared_hidden_size + config.base_char_vocab_size, config.shared_hidden_size), nn.Sigmoid())
            diacritic_head_input_size = config.shared_hidden_size
            logger.info("'gate' conditioning for diacritic head.")
        else: # 'none' or unknown
            if config.conditioning_method != 'none': logger.warning(f"Conditioning '{config.conditioning_method}' unknown, using 'none'."); self.config.conditioning_method = 'none'
            diacritic_head_input_size = config.shared_hidden_size; logger.info("'none' conditioning.")

        # Diacritic Enhancement modules (from your previous code)
        self.visual_diacritic_attention = None
        self.character_diacritic_compatibility = None
        self.few_shot_diacritic_adapter = None
        if config.use_visual_diacritic_attention: self.visual_diacritic_attention = VisualDiacriticAttention(diacritic_head_input_size, config.diacritic_vocab_size); logger.info("Visual Diacritic Attn on.")
        if config.use_character_diacritic_compatibility: self.character_diacritic_compatibility = CharacterDiacriticCompatibility(config.base_char_vocab_size, config.diacritic_vocab_size, config.shared_hidden_size, self.base_char_vocab, self.diacritic_vocab); logger.info("Char-Diac Compat on.")
        if config.use_few_shot_diacritic_adapter: self.few_shot_diacritic_adapter = FewShotDiacriticAdapter(diacritic_head_input_size, config.diacritic_vocab_size, config.num_few_shot_prototypes); logger.info("Few-Shot Adapter on.")
        
        self.diacritic_classifier = nn.Linear(diacritic_head_input_size, config.diacritic_vocab_size)
        logger.info(f"Diacritic Classifier Head (In: {diacritic_head_input_size}, Out: {config.diacritic_vocab_size})")

        final_combiner_input_size = config.shared_hidden_size
        self.final_classifier = nn.Linear(final_combiner_input_size, config.combined_char_vocab_size)
        logger.info(f"Final Combined Classifier Head (In: {final_combiner_input_size}, Out: {config.combined_char_vocab_size})")

        self._init_weights()

    def _init_weights(self):
        logger.debug("Initializing weights for new layers (Transformer, Shared, Heads, Proj, Gate)...")
        # Vision Encoder is pretrained
        # Fusion Projection
        if self.fusion_projection: nn.init.xavier_uniform_(self.fusion_projection.weight); nn.init.zeros_(self.fusion_projection.bias)
        if hasattr(self, 'pre_fusion_projections') and self.pre_fusion_projections:
            for proj in self.pre_fusion_projections: nn.init.xavier_uniform_(proj.weight); nn.init.zeros_(proj.bias)
        if self.fusion_bilinear: nn.init.xavier_uniform_(self.fusion_bilinear.weight); nn.init.zeros_(self.fusion_bilinear.bias)
        # Positional Encoding (Learned)
        if isinstance(self.pos_encoder, LearnedPositionalEncoding1D): nn.init.normal_(self.pos_encoder.pos_embedding.weight, mean=0, std=0.02)
        # Transformer Encoder: PyTorch's default init for TransformerEncoderLayer is generally good (Xavier for linears, etc.)
        # Shared Layers
        for layer in self.shared_layer:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        # Heads
        for head in [self.base_classifier, self.diacritic_classifier, self.final_classifier]:
            nn.init.xavier_uniform_(head.weight); nn.init.zeros_(head.bias)
        # Gate & Condition Proj
        if self.diacritic_gate:
             for layer in self.diacritic_gate:
                  if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        if self.diacritic_condition_proj:
             for layer in self.diacritic_condition_proj:
                  if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        # Init new diacritic enhancement modules if they are actual nn.Module subclasses
        for mod_name in ['visual_diacritic_attention', 'character_diacritic_compatibility', 'few_shot_diacritic_adapter']:
            module = getattr(self, mod_name, None)
            if module and isinstance(module, nn.Module):
                 for name, param in module.named_parameters():
                      if param.ndim > 1: nn.init.xavier_uniform_(param) # Simple init for linear/conv
                      elif 'bias' in name: nn.init.zeros_(param)


    def forward(self, pixel_values, labels=None, label_lengths=None):
        # 1. Vision Encoder
        encoder_outputs = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        all_hidden_states = encoder_outputs.hidden_states

        # 2. Select and Fuse Features
        num_encoder_layers = len(all_hidden_states) -1
        valid_indices = [idx if idx >=0 else num_encoder_layers + 1 + idx for idx in self.config.vision_encoder_layer_indices]
        valid_indices = [i for i in valid_indices if 0 < i <= num_encoder_layers + 1]
        if not valid_indices: features_to_fuse = [all_hidden_states[-1]]
        else: features_to_fuse = [all_hidden_states[i] for i in valid_indices]

        fused_features = None
        if self.config.feature_fusion_method == "concat_proj" and len(features_to_fuse) > 1 and self.fusion_projection:
            concatenated_features = torch.cat(features_to_fuse, dim=-1)
            fused_features = self.fusion_projection(concatenated_features)
        elif self.config.feature_fusion_method == "add" and len(features_to_fuse) > 1:
            if hasattr(self, 'pre_fusion_projections') and self.pre_fusion_projections:
                projected_features = [self.pre_fusion_projections[i](feat) for i, feat in enumerate(features_to_fuse)]
                fused_features = torch.stack(projected_features, dim=0).mean(dim=0)
            else: fused_features = torch.stack(features_to_fuse, dim=0).mean(dim=0)
        elif self.config.feature_fusion_method == "bilinear" and len(features_to_fuse) == 2 and self.fusion_bilinear:
             fused_features = self.fusion_bilinear(features_to_fuse[0], features_to_fuse[1])
        else: # 'none' or fallback
            last_layer_features = features_to_fuse[-1]
            if self.fusion_projection and self.config.feature_fusion_method != "concat_proj": # Project if only last layer AND mismatch
                 fused_features = self.fusion_projection(last_layer_features)
            elif self.config.feature_fusion_method == "concat_proj" and len(features_to_fuse)==1 and self.fusion_projection: # single layer selected but proj exists
                 fused_features = self.fusion_projection(last_layer_features) # This case means last_layer_features should match proj input
            else: fused_features = last_layer_features

        if fused_features is None: # Safety fallback
            logger.error("Fused features are None! Using last vision encoder layer directly.")
            fused_features = all_hidden_states[-1]
            # If dimensions still don't match transformer_d_model, it will error.
            # Ensure transformer_d_model is set to encoder_hidden_size if no fusion/projection.

        # 3. Positional Encoding
        features_with_pos = self.pos_encoder(fused_features)

        # 4. Transformer Encoder Layers
        # TODO: src_key_padding_mask if fused_features can be padded
        transformer_output = self.transformer_encoder(features_with_pos)

        # 5. Shared Feature Layer
        shared_features = self.shared_layer(transformer_output)

        # 6. Hierarchical Heads
        base_logits = self.base_classifier(shared_features)
        diacritic_input_features = shared_features
        if self.config.conditioning_method == 'concat_proj':
            concat_features = torch.cat((shared_features, base_logits), dim=-1)
            diacritic_input_features = self.diacritic_condition_proj(concat_features)
        elif self.config.conditioning_method == 'gate' and self.diacritic_gate is not None:
            gate_input = torch.cat((shared_features, base_logits), dim=-1)
            diacritic_gate_values = self.diacritic_gate(gate_input)
            diacritic_input_features = shared_features * diacritic_gate_values

        # Standard diacritic logits
        standard_diacritic_logits = self.diacritic_classifier(diacritic_input_features)
        # Enhanced diacritic logits
        current_diacritic_logits = standard_diacritic_logits
        if self.visual_diacritic_attention: current_diacritic_logits = current_diacritic_logits + self.visual_diacritic_attention(diacritic_input_features)
        if self.character_diacritic_compatibility: current_diacritic_logits = current_diacritic_logits + self.character_diacritic_compatibility(base_logits, shared_features if self.config.conditioning_method != 'none' else None)
        if self.few_shot_diacritic_adapter: current_diacritic_logits = current_diacritic_logits + self.few_shot_diacritic_adapter(diacritic_input_features)
        diacritic_logits = current_diacritic_logits


        # 7. Final Combined Classifier
        final_input_features = shared_features
        final_logits = self.final_classifier(final_input_features)

        # --- CTC Loss ---
        log_probs = final_logits.log_softmax(dim=2).permute(1, 0, 2)
        loss = None
        if labels is not None and label_lengths is not None:
            device = final_logits.device; time_steps = log_probs.size(0); batch_size_actual = log_probs.size(1)
            input_lengths = torch.full((batch_size_actual,), time_steps, dtype=torch.long, device=device)
            labels_device = labels.to(device); label_lengths_device = label_lengths.to(device)
            input_lengths_clamped = torch.clamp(input_lengths, max=time_steps)
            label_lengths_clamped = torch.clamp(label_lengths_device, max=labels_device.size(1))
            try:
                ctc_loss_fn = nn.CTCLoss(blank=self.config.blank_idx, reduction='mean', zero_infinity=True)
                loss = ctc_loss_fn(log_probs, labels_device, input_lengths_clamped, label_lengths_clamped)
            except Exception as e: logger.error(f"CTC loss error: {e}"); loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'loss': loss, 'logits': final_logits,
            'base_logits': base_logits, 'diacritic_logits': diacritic_logits,
        }

    # --- save_pretrained and from_pretrained methods (same as HierarchicalCtcV2OcrModel) ---
    def save_pretrained(self, save_directory, **kwargs):
        logger.info(f"Saving {self.__class__.__name__} model to: {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self, 'base_char_vocab'): self.config.base_char_vocab = self.base_char_vocab
        if hasattr(self, 'diacritic_vocab'): self.config.diacritic_vocab = self.diacritic_vocab
        if hasattr(self, 'combined_char_vocab'): self.config.combined_char_vocab = self.combined_char_vocab
        self.config.base_char_vocab_size = len(getattr(self, 'base_char_vocab', []))
        self.config.diacritic_vocab_size = len(getattr(self, 'diacritic_vocab', []))
        self.config.combined_char_vocab_size = len(getattr(self, 'combined_char_vocab', []))
        if hasattr(self.vision_encoder, 'config'): self.config.vision_encoder_config = self.vision_encoder.config
        self.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, "pytorch_model.bin"); torch.save(self.state_dict(), output_model_file)
        try:
            if hasattr(self, 'processor') and self.processor: self.processor.save_pretrained(save_directory)
        except Exception as e: logger.error(f"Failed to save processor: {e}")
        logger.info(f"Model components saved.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        logger.info(f"Loading {cls.__name__} from: {pretrained_model_name_or_path}")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json"); loaded_config = None
        if config is not None and isinstance(config, cls.config_class): loaded_config = config; logger.info("Using provided config.")
        elif os.path.exists(config_path): loaded_config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs); logger.info(f"Loading config from: {config_path}")
        else:
            logger.warning(f"Config not found at {config_path}. Initializing new from kwargs.")
            if not all(k in kwargs for k in ['base_char_vocab', 'diacritic_vocab', 'combined_char_vocab']): raise ValueError("All vocabs required in kwargs w/o config.json.")
            if 'vision_encoder_name' not in kwargs: kwargs['vision_encoder_name'] = cls.config_class().vision_encoder_name
            loaded_config = cls.config_class(**kwargs)
        if 'base_char_vocab' in kwargs and kwargs['base_char_vocab']: loaded_config.base_char_vocab = kwargs['base_char_vocab']; loaded_config.base_char_vocab_size = len(kwargs['base_char_vocab'])
        if 'diacritic_vocab' in kwargs and kwargs['diacritic_vocab']: loaded_config.diacritic_vocab = kwargs['diacritic_vocab']; loaded_config.diacritic_vocab_size = len(kwargs['diacritic_vocab'])
        if 'combined_char_vocab' in kwargs and kwargs['combined_char_vocab']: loaded_config.combined_char_vocab = kwargs['combined_char_vocab']; loaded_config.combined_char_vocab_size = len(kwargs['combined_char_vocab'])
        model = cls(loaded_config)
        state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"); is_loading_base_model = not os.path.exists(config_path)
        if os.path.exists(state_dict_path) and not is_loading_base_model:
            logger.info(f"Loading state dict from: {state_dict_path}")
            try: state_dict = torch.load(state_dict_path, map_location="cpu"); load_result = model.load_state_dict(state_dict, strict=False); logger.info(f"Loaded state. Miss:{load_result.missing_keys}, Unexp:{load_result.unexpected_keys}")
            except Exception as e: logger.error(f"Error loading state dict: {e}")
        elif is_loading_base_model: logger.info(f"Loading base weights only.")
        else: logger.warning(f"State dict not found. Using base + random.")
        model.base_char_vocab = loaded_config.base_char_vocab; model.diacritic_vocab = loaded_config.diacritic_vocab; model.combined_char_vocab = loaded_config.combined_char_vocab
        return model