# model/diacritic_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class VisualDiacriticAttention(nn.Module):
    """
    Visual attention mechanism that focuses on the areas where diacritics typically appear.
    Creates three attention regions: above, middle, and below the base character.
    Each region has a specialized classifier for diacritics that commonly appear in that position.
    """
    def __init__(self, feature_dim, diacritic_vocab_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.diacritic_vocab_size = diacritic_vocab_size
        
        # Position encoder to generate attention weights for three regions
        self.position_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 3)  # 3 attention maps: above, middle, below
        )
        
        # Define region-specific diacritic classifiers
        # Each classifier focuses on diacritics that typically appear in that region
        self.region_classifiers = nn.ModuleList([
            # Above diacritics (acute, grave, hook, tilde)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            ),
            # Middle diacritics (circumflex, breve, horn)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            ),
            # Below diacritics (dot below)
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, diacritic_vocab_size)
            )
        ])
        
        # Final fusion layer to combine region-specific predictions
        self.output_fusion = nn.Linear(diacritic_vocab_size * 3, diacritic_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    # def forward(self, features):
    #     """
    #     Apply visual diacritic attention to input features.
        
    #     Args:
    #         features: Tensor of shape [batch_size, seq_length, feature_dim]
    #                 Typically the output of shared feature layers
        
    #     Returns:
    #         diacritic_logits: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
    #     """
    #     batch_size, seq_length, _ = features.shape
        
    #     # Generate position attention weights
    #     # [batch_size, seq_length, 3]
    #     position_logits = self.position_encoder(features)
    #     position_weights = F.softmax(position_logits, dim=-1)
        
    #     # Apply each specialized classifier to all positions
    #     region_logits = []
    #     for i, classifier in enumerate(self.region_classifiers):
    #         # [batch_size, seq_length, diacritic_vocab_size]
    #         region_output = classifier(features)
            
    #         # Weight by position attention
    #         # [batch_size, seq_length, 1] * [batch_size, seq_length, diacritic_vocab_size]
    #         weighted_output = position_weights[:, :, i:i+1] * region_output
    #         region_logits.append(weighted_output)
        
    #     # Option 1: Sum the weighted outputs from different regions
    #     # diacritic_logits = sum(region_logits)
        
    #     # Option 2: Concatenate and fuse with a linear layer (more expressive)
    #     # [batch_size, seq_length, diacritic_vocab_size*3]
    #     concatenated_logits = torch.cat(region_logits, dim=-1)
    #     # [batch_size, seq_length, diacritic_vocab_size]
    #     diacritic_logits = self.output_fusion(concatenated_logits)
        
    #     return diacritic_logits
    def forward(self, features, return_attention_weights=False): # Added return_attention_weights
            batch_size, seq_length, _ = features.shape
            
            position_logits = self.position_encoder(features)
            # position_weights are [batch_size, seq_length, 3] (for above, middle, below)
            position_weights = F.softmax(position_logits, dim=-1) 
            
            region_logits_list = [] # Renamed from region_logits to avoid confusion
            for i, classifier in enumerate(self.region_classifiers):
                region_output = classifier(features)
                weighted_output = position_weights[:, :, i:i+1] * region_output
                region_logits_list.append(weighted_output)
            
            concatenated_logits = torch.cat(region_logits_list, dim=-1)
            final_diacritic_logits = self.output_fusion(concatenated_logits)
            
            if return_attention_weights:
                return final_diacritic_logits, position_weights
            return final_diacritic_logits
    
class CharacterDiacriticCompatibility(nn.Module):
    """
    Explicit Character-Diacritic Compatibility Matrix.
    Models which diacritics are compatible with which base characters,
    e.g., vowels can have tone marks but consonants cannot.
    """
    def __init__(self, base_vocab_size, diacritic_vocab_size, shared_dim=None, 
                 base_char_vocab=None, diacritic_vocab=None):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.diacritic_vocab_size = diacritic_vocab_size
        self.base_char_vocab = base_char_vocab if base_char_vocab else []
        self.diacritic_vocab = diacritic_vocab if diacritic_vocab else []
        
        self.compatibility_matrix = nn.Parameter(
            torch.randn(base_vocab_size, diacritic_vocab_size) * 0.01 # Initialize with small random values
        )
        
        self.use_shared_features = shared_dim is not None
        if self.use_shared_features:
            self.compatibility_predictor = nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.LayerNorm(shared_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim // 2, diacritic_vocab_size)
            )
        
        if self.base_char_vocab and self.diacritic_vocab:
            self._initialize_compatibility()
        else:
            logger.warning("Base/Diacritic vocabs not provided for CharacterDiacriticCompatibility init. Skipping detailed linguistic prior initialization.")

    def _initialize_compatibility(self):
        """Initialize the compatibility matrix with detailed Vietnamese linguistic knowledge."""
        if not self.base_char_vocab or not self.diacritic_vocab:
            logger.warning("Cannot initialize compatibility matrix: base_char_vocab or diacritic_vocab is empty.")
            return

        with torch.no_grad():
            self.compatibility_matrix.fill_(-3.0) 

            def get_idx(vocab_list_original, item_name_target_lower, not_found_val=-1):
                for i, item in enumerate(vocab_list_original):
                    if isinstance(item, str) and item.lower() == item_name_target_lower:
                        return i
                return not_found_val

            no_diac_idx = get_idx(self.diacritic_vocab, 'no_diacritic')
            blank_diac_idx = get_idx(self.diacritic_vocab, '<blank>')
            unk_diac_idx = get_idx(self.diacritic_vocab, '<unk>')
            
            pure_tone_names_lower = ['acute', 'grave', 'hook', 'tilde', 'dot']
            pure_tone_indices = [get_idx(self.diacritic_vocab, n) for n in pure_tone_names_lower]
            pure_tone_indices = [i for i in pure_tone_indices if i != -1]

            MODIFIERS_MAP = {
                'breve': get_idx(self.diacritic_vocab, 'breve'),
                'horn': get_idx(self.diacritic_vocab, 'horn'),
                'circumflex': get_idx(self.diacritic_vocab, 'circumflex'),
                'stroke': get_idx(self.diacritic_vocab, 'stroke')
            }
            MODIFIERS_MAP = {k: v for k, v in MODIFIERS_MAP.items() if v != -1}

            VI_PLAIN_VOWELS_LOWER = ['a', 'e', 'i', 'o', 'u', 'y']
            # Now 'đ' is not pre-diacriticized, so it should be in the general consonant list if it behaves like one.
            # Or, if it has unique behavior, it's handled as a special case outside this list.
            # For now, let's assume 'd' is special (for stroke), and 'đ' (if present as a base char) is a standard consonant.
            VI_CONSONANTS_LOWER = ['b', 'c', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'f', 'j', 'w', 'z'] # Added 'đ' here
            VI_D_LOWER = 'd' # This is the base 'd' that can take a stroke
            
            # PRE_DIACRITICIZED_BASE_LOWER = ['đ'] # REMOVED 'đ' from here
            PRE_DIACRITICIZED_BASE_LOWER = [] # List of base chars that are *already modified* and should ONLY take 'no_diacritic'

            logger.info(f"Initializing compatibility matrix with priors: no_diac_idx={no_diac_idx}, blank_diac_idx={blank_diac_idx}")
            logger.info(f"Pure tone indices found: {pure_tone_indices}")
            logger.info(f"Modifiers map (name_lower -> index): {MODIFIERS_MAP}")

            for base_idx, base_char_orig in enumerate(self.base_char_vocab):
                if not isinstance(base_char_orig, str): continue 
                base_char_lower = base_char_orig.lower()

                if base_char_lower == '<blank>':
                    if blank_diac_idx != -1: self.compatibility_matrix[base_idx, blank_diac_idx] = 5.0
                    continue 
                
                if base_char_lower == '<unk>':
                    if unk_diac_idx != -1: self.compatibility_matrix[base_idx, unk_diac_idx] = 3.0
                    if no_diac_idx != -1: self.compatibility_matrix[base_idx, no_diac_idx] = 2.0
                    continue

                # Default compatibility with 'no_diacritic' for all "normal" characters
                if no_diac_idx != -1:
                    self.compatibility_matrix[base_idx, no_diac_idx] = 2.5 

                # Handle characters that are truly pre-diacriticized in the base vocab (if any remain)
                if base_char_lower in PRE_DIACRITICIZED_BASE_LOWER: # This list is now empty, but kept for structure
                    # These are already modified. They should only be compatible with 'no_diacritic'.
                    # (Handled by the general rule above, others remain -3.0)
                    continue

                # Handle base 'd' (which can take 'stroke' to become 'đ')
                if base_char_lower == VI_D_LOWER: # This is for 'd', not 'đ'
                    stroke_idx = MODIFIERS_MAP.get('stroke', -1)
                    if stroke_idx != -1:
                        self.compatibility_matrix[base_idx, stroke_idx] = 3.5 # d + stroke -> đ
                    # 'd' itself is a consonant, so it shouldn't take other tones/modifiers directly.
                    # (Handled by falling through to consonant logic if not d + stroke)
                    # The primary purpose here is to allow 'd' + 'stroke'. Other diacritics for 'd' will be -3.0.
                    continue # Important to continue to prevent 'd' from being treated as a regular consonant below for other diacritics

                # Handle other consonants (including 'đ' now, if it's in VI_CONSONANTS_LOWER)
                if base_char_lower in VI_CONSONANTS_LOWER:
                    # Consonants (including 'đ') are compatible with 'no_diacritic' (set above).
                    # Generally, Vietnamese consonants (including 'đ') do not take tones or modifiers.
                    # So, all other diacritics for them remain strongly negative (-3.0).
                    continue

                # --- Logic for Plain Vowels ---
                if base_char_lower in VI_PLAIN_VOWELS_LOWER:
                    # 1. Pure Tones for all plain vowels
                    for tone_idx in pure_tone_indices:
                        self.compatibility_matrix[base_idx, tone_idx] = 3.0

                    # 2. Pure Modifiers (specific to vowels)
                    can_take_modifier_indices = [] 
                    if base_char_lower == 'a':
                        if MODIFIERS_MAP.get('breve', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['breve']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['breve'])
                        if MODIFIERS_MAP.get('circumflex', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['circumflex']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['circumflex'])
                    elif base_char_lower == 'e':
                        if MODIFIERS_MAP.get('circumflex', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['circumflex']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['circumflex'])
                    elif base_char_lower == 'o':
                        if MODIFIERS_MAP.get('circumflex', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['circumflex']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['circumflex'])
                        if MODIFIERS_MAP.get('horn', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['horn']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['horn'])
                    elif base_char_lower == 'u':
                        if MODIFIERS_MAP.get('horn', -1) != -1:
                            self.compatibility_matrix[base_idx, MODIFIERS_MAP['horn']] = 3.0
                            can_take_modifier_indices.append(MODIFIERS_MAP['horn'])
                    
                    # 3. Combined Diacritics (Modifier_Tone)
                    for combined_diac_idx, combined_diac_str_orig in enumerate(self.diacritic_vocab):
                        if not isinstance(combined_diac_str_orig, str): continue
                        combined_diac_str_lower = combined_diac_str_orig.lower()
                        
                        parts = combined_diac_str_lower.split('_')
                        if len(parts) == 2: 
                            mod_part_name_lower, tone_part_name_lower = parts[0], parts[1]
                            is_valid_tone_part = tone_part_name_lower in pure_tone_names_lower
                            modifier_part_idx_in_vocab = MODIFIERS_MAP.get(mod_part_name_lower, -1)

                            if is_valid_tone_part and \
                               modifier_part_idx_in_vocab != -1 and \
                               modifier_part_idx_in_vocab in can_take_modifier_indices:
                                self.compatibility_matrix[base_idx, combined_diac_idx] = 3.0
                # else: Covers numbers, symbols etc. They are compatible with 'no_diacritic' (set earlier), 
                # and all other diacritics remain -3.0.

            logger.info("Completed initialization of compatibility matrix with detailed Vietnamese linguistic priors (đ treated as a regular consonant).")

    def forward(self, base_logits, shared_features=None):
        """
        Apply compatibility constraints between base characters and diacritics.
        
        Args:
            base_logits: Tensor of shape [batch_size, seq_length, base_vocab_size]
                        Logits from the base character classifier
            shared_features: Optional tensor of shape [batch_size, seq_length, shared_dim]
                            Shared features for additional compatibility prediction
        
        Returns:
            compatibility_bias: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
                               Bias to add to diacritic logits based on compatibility
        """
        # Convert base logits to probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Calculate compatibility bias using matrix multiplication
        # [batch_size, seq_length, base_vocab_size] × [base_vocab_size, diacritic_vocab_size]
        # -> [batch_size, seq_length, diacritic_vocab_size]
        compatibility_bias = torch.matmul(base_probs, self.compatibility_matrix)
        if self.training:
            # Only do this during training
            direct_term = 0.001 * F.relu(self.compatibility_matrix).mean()
            # Add this term in a way that maintains the gradient flow
            compatibility_bias = compatibility_bias + direct_term.expand_as(compatibility_bias)
        
        # If using shared features, add additional compatibility prediction
        if self.use_shared_features and shared_features is not None:
            additional_bias = self.compatibility_predictor(shared_features)
            compatibility_bias = compatibility_bias + additional_bias
        
        return compatibility_bias, self.compatibility_matrix


class FewShotDiacriticAdapter(nn.Module):
    """
    Few-Shot Diacritic Learning adapter for rare diacritic combinations.
    Uses a prototype-based approach to generalize from few examples.
    """
    def __init__(self, feature_dim, diacritic_vocab_size, num_prototypes=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.diacritic_vocab_size = diacritic_vocab_size
        self.num_prototypes = num_prototypes
        
        # Learnable prototypes for each diacritic class
        # These will be updated during training to represent typical feature patterns
        self.prototypes = nn.Parameter(
            torch.randn(diacritic_vocab_size, num_prototypes, feature_dim)
        )
        
        # Scaling factor for prototype matching (temperature parameter)
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
        # Optional: attention mechanism to focus on relevant feature dimensions
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(diacritic_vocab_size, diacritic_vocab_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize prototypes to have unit norm
        with torch.no_grad():
            normalized_prototypes = F.normalize(self.prototypes, p=2, dim=-1)
            self.prototypes.copy_(normalized_prototypes)
    
    def forward(self, features):
        """
        Apply few-shot diacritic learning to input features.
        
        Args:
            features: Tensor of shape [batch_size, seq_length, feature_dim]
                    Typically the output of shared feature layers
        
        Returns:
            diacritic_logits: Tensor of shape [batch_size, seq_length, diacritic_vocab_size]
        """
        batch_size, seq_length, _ = features.shape
        
        # Apply feature attention
        attention_weights = self.feature_attention(features)
        weighted_features = features * attention_weights
        
        # Normalize features for cosine similarity
        normalized_features = F.normalize(weighted_features, p=2, dim=-1)
        
        # Calculate similarity to each prototype
        # For efficiency, reshape and compute all similarities at once
        # Reshape prototypes: [diacritic_vocab_size, num_prototypes, feature_dim]
        # -> [diacritic_vocab_size * num_prototypes, feature_dim]
        flat_prototypes = self.prototypes.view(-1, self.feature_dim)
        
        # Reshape features: [batch_size, seq_length, feature_dim]
        # -> [batch_size * seq_length, feature_dim]
        flat_features = normalized_features.view(-1, self.feature_dim)
        
        # Compute similarities
        # [batch_size * seq_length, diacritic_vocab_size * num_prototypes]
        similarities = torch.matmul(flat_features, flat_prototypes.transpose(0, 1))
        
        # Reshape similarities to [batch_size, seq_length, diacritic_vocab_size, num_prototypes]
        similarities = similarities.view(batch_size, seq_length, self.diacritic_vocab_size, self.num_prototypes)
        
        # Take max similarity for each class (max-pooling over prototypes)
        # [batch_size, seq_length, diacritic_vocab_size]
        max_similarities, _ = torch.max(similarities, dim=-1)
        
        # Apply temperature scaling
        scaled_similarities = max_similarities * self.temperature
        
        # Optional: Apply output projection for better expressivity
        diacritic_logits = self.output_projection(scaled_similarities)
        
        return diacritic_logits