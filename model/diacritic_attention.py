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
            # Start with STRONG negative bias for everything
            self.compatibility_matrix.fill_(-5.0)  # Even more negative to ensure blocking

            def get_idx(vocab_list_original, item_name_target_lower, not_found_val=-1):
                for i, item in enumerate(vocab_list_original):
                    if isinstance(item, str) and item.lower() == item_name_target_lower:
                        return i
                return not_found_val

            # Get special indices
            no_diac_idx = get_idx(self.diacritic_vocab, 'no_diacritic')
            blank_diac_idx = get_idx(self.diacritic_vocab, '<blank>')
            unk_diac_idx = get_idx(self.diacritic_vocab, '<unk>')
            
            # Pure tone mark indices
            pure_tone_names_lower = ['acute', 'grave', 'hook', 'tilde', 'dot']
            pure_tone_indices = [get_idx(self.diacritic_vocab, n) for n in pure_tone_names_lower]
            pure_tone_indices = [i for i in pure_tone_indices if i != -1]

            # Modifier indices
            MODIFIERS_MAP = {
                'breve': get_idx(self.diacritic_vocab, 'breve'),
                'horn': get_idx(self.diacritic_vocab, 'horn'),
                'circumflex': get_idx(self.diacritic_vocab, 'circumflex'),
                'stroke': get_idx(self.diacritic_vocab, 'stroke')
            }
            MODIFIERS_MAP = {k: v for k, v in MODIFIERS_MAP.items() if v != -1}

            # 🎯 CORRECTED: Proper Vietnamese character classification
            VI_VOWELS_LOWER = ['a', 'e', 'i', 'o', 'u', 'y']  # Pure vowels that can take diacritics
            VI_CONSONANTS_LOWER = [
                'b', 'c', 'd', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 
                'f', 'j', 'w', 'z'  # All consonants - should ONLY take 'no_diacritic'
            ]
            VI_SPECIAL_D = 'd'  # Can take stroke to become 'đ'

            logger.info(f"Initializing compatibility with linguistic rules:")
            logger.info(f"  - Vowels that can take diacritics: {VI_VOWELS_LOWER}")
            logger.info(f"  - Consonants (no diacritics): {VI_CONSONANTS_LOWER}")

            for base_idx, base_char_orig in enumerate(self.base_char_vocab):
                if not isinstance(base_char_orig, str): 
                    continue 
                base_char_lower = base_char_orig.lower()

                # Handle special tokens
                if base_char_lower == '<blank>':
                    if blank_diac_idx != -1: 
                        self.compatibility_matrix[base_idx, blank_diac_idx] = 5.0
                    continue 
                
                if base_char_lower == '<unk>':
                    if unk_diac_idx != -1: 
                        self.compatibility_matrix[base_idx, unk_diac_idx] = 3.0
                    if no_diac_idx != -1: 
                        self.compatibility_matrix[base_idx, no_diac_idx] = 2.0
                    continue

                # 🔥 RULE 1: ALL characters can have 'no_diacritic'
                if no_diac_idx != -1:
                    self.compatibility_matrix[base_idx, no_diac_idx] = 3.0

                # 🔥 RULE 2: CONSONANTS - ONLY 'no_diacritic' (except 'd' + stroke)
                if base_char_lower in VI_CONSONANTS_LOWER:
                    if base_char_lower == VI_SPECIAL_D:
                        # Special case: 'd' can take stroke to become 'đ'
                        stroke_idx = MODIFIERS_MAP.get('stroke', -1)
                        if stroke_idx != -1:
                            self.compatibility_matrix[base_idx, stroke_idx] = 3.5
                            logger.debug(f"Allowing 'd' + stroke → 'đ'")
                    
                    # For ALL consonants (including 'd'), only 'no_diacritic' is allowed
                    # All tone marks and other modifiers remain at -5.0 (strongly forbidden)
                    logger.debug(f"Consonant '{base_char_lower}' → only 'no_diacritic' allowed")
                    continue

                # 🔥 RULE 3: VOWELS - Can take tone marks and specific modifiers
                if base_char_lower in VI_VOWELS_LOWER:
                    # All pure tones are allowed for all vowels
                    for tone_idx in pure_tone_indices:
                        self.compatibility_matrix[base_idx, tone_idx] = 4.0

                    # Vowel-specific modifiers
                    vowel_modifiers = []
                    if base_char_lower == 'a':
                        vowel_modifiers = ['breve', 'circumflex']
                    elif base_char_lower == 'e':
                        vowel_modifiers = ['circumflex']
                    elif base_char_lower == 'o':
                        vowel_modifiers = ['circumflex', 'horn']
                    elif base_char_lower == 'u':
                        vowel_modifiers = ['horn']

                    # Apply vowel-specific modifiers
                    for modifier_name in vowel_modifiers:
                        modifier_idx = MODIFIERS_MAP.get(modifier_name, -1)
                        if modifier_idx != -1:
                            self.compatibility_matrix[base_idx, modifier_idx] = 4.0

                    # Combined diacritics (modifier + tone)
                    for combined_diac_idx, combined_diac_str_orig in enumerate(self.diacritic_vocab):
                        if not isinstance(combined_diac_str_orig, str): 
                            continue
                        combined_diac_str_lower = combined_diac_str_orig.lower()
                        
                        parts = combined_diac_str_lower.split('_')
                        if len(parts) == 2: 
                            mod_part, tone_part = parts[0], parts[1]
                            
                            # Check if this combination is valid for this vowel
                            if (tone_part in pure_tone_names_lower and 
                                mod_part in vowel_modifiers):
                                self.compatibility_matrix[base_idx, combined_diac_idx] = 4.0

                    logger.debug(f"Vowel '{base_char_lower}' → allowed modifiers: {vowel_modifiers}")
                    continue

                # 🔥 RULE 4: Numbers, symbols, punctuation - only 'no_diacritic'
                # Everything else (numbers, symbols, punctuation) only gets 'no_diacritic'
                # The matrix is already initialized to -5.0, and 'no_diacritic' is set to 3.0 above
                logger.debug(f"Non-letter '{base_char_lower}' → only 'no_diacritic' allowed")

            logger.info("Compatibility matrix initialization complete with strict Vietnamese rules")
            
            # 🔍 Debug: Print some examples
            for test_char in ['a', 'b', 'g', 'o']:
                if test_char in [c.lower() for c in self.base_char_vocab]:
                    char_idx = next(i for i, c in enumerate(self.base_char_vocab) if c.lower() == test_char)
                    char_row = self.compatibility_matrix[char_idx]
                    max_idx = torch.argmax(char_row)
                    min_idx = torch.argmin(char_row)
                    logger.info(f"'{test_char}' → best: {self.diacritic_vocab[max_idx]} ({char_row[max_idx]:.1f}), "
                            f"worst: {self.diacritic_vocab[min_idx]} ({char_row[min_idx]:.1f})")

    def _create_ideal_compatibility_matrix(self):
        """Create the ideal compatibility matrix based on Vietnamese linguistic rules"""
        
        ideal_matrix = torch.full(
            (len(self.base_char_vocab), len(self.diacritic_vocab)), 
            -5.0  # Strong negative default
        )
        
        # Apply the same initialization logic as _initialize_compatibility
        # but return as tensor instead of updating self.compatibility_matrix
        
        def get_idx(vocab_list, target_name):
            try:
                return vocab_list.index(target_name.lower())
            except (ValueError, AttributeError):
                return -1
        
        no_diac_idx = get_idx(self.diacritic_vocab, 'no_diacritic')
        pure_tone_names = ['acute', 'grave', 'hook', 'tilde', 'dot']
        pure_tone_indices = [get_idx(self.diacritic_vocab, tone) for tone in pure_tone_names]
        pure_tone_indices = [i for i in pure_tone_indices if i != -1]
        
        VI_VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']
        VI_CONSONANTS = ['b', 'c', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'f', 'j', 'z']
        VI_SPECIAL_D = 'd'
        MODIFIERS_MAP = {
                'breve': get_idx(self.diacritic_vocab, 'breve'),
                'horn': get_idx(self.diacritic_vocab, 'horn'),
                'circumflex': get_idx(self.diacritic_vocab, 'circumflex'),
                'stroke': get_idx(self.diacritic_vocab, 'stroke')
            }
        MODIFIERS_MAP = {k: v for k, v in MODIFIERS_MAP.items() if v != -1}

        for base_idx, base_char in enumerate(self.base_char_vocab):
            if not isinstance(base_char, str):
                continue
            base_lower = base_char.lower()
            
            # Rule 1: All characters can have 'no_diacritic'
            if no_diac_idx != -1:
                ideal_matrix[base_idx, no_diac_idx] = 3.0
            
            # Rule 2: Consonants - ONLY 'no_diacritic'
            if base_lower in VI_CONSONANTS:
                if base_lower == VI_SPECIAL_D:
                    # Special case: 'd' can take stroke to become 'đ'
                    stroke_idx = MODIFIERS_MAP.get('stroke', -1)
                    if stroke_idx != -1:
                        ideal_matrix[base_idx, stroke_idx] = 3.5
                        logger.debug(f"Allowing 'd' + stroke → 'đ'")
                    
                # For ALL consonants (including 'd'), only 'no_diacritic' is allowed
                # All tone marks and other modifiers remain at -5.0 (strongly forbidden)
                logger.debug(f"Consonant '{base_lower}' → only 'no_diacritic' allowed")
                continue

            # Rule 3: Vowels - can take tone marks
            if base_lower in VI_VOWELS:
                for tone_idx in pure_tone_indices:
                    ideal_matrix[base_idx, tone_idx] = 4.0
                # Vowel-specific modifiers
                vowel_modifiers = []
                if base_lower == 'a':
                    vowel_modifiers = ['breve', 'circumflex']
                elif base_lower == 'e':
                    vowel_modifiers = ['circumflex']
                elif base_lower == 'o':
                    vowel_modifiers = ['circumflex', 'horn']
                elif base_lower == 'u':
                    vowel_modifiers = ['horn']

                # Apply vowel-specific modifiers
                for modifier_name in vowel_modifiers:
                    modifier_idx = MODIFIERS_MAP.get(modifier_name, -1)
                    if modifier_idx != -1:
                        ideal_matrix[base_idx, modifier_idx] = 4.0

                # Combined diacritics (modifier + tone)
                for combined_diac_idx, combined_diac_str_orig in enumerate(self.diacritic_vocab):
                    if not isinstance(combined_diac_str_orig, str): 
                        continue
                    combined_diac_str_lower = combined_diac_str_orig.lower()
                    
                    parts = combined_diac_str_lower.split('_')
                    if len(parts) == 2: 
                        mod_part, tone_part = parts[0], parts[1]
                        
                        # Check if this combination is valid for this vowel
                        if (tone_part in pure_tone_names and 
                            mod_part in vowel_modifiers):
                            ideal_matrix[base_idx, combined_diac_idx] = 4.0

                logger.debug(f"Vowel '{base_lower}' → allowed modifiers: {vowel_modifiers}")
                continue
        

        
        return ideal_matrix

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

    def get_linguistic_regularization_loss(self, strength=1.0):
        """Add regularization to keep compatibility matrix linguistically correct"""
        if not hasattr(self, '_ideal_compatibility_matrix'):
            # Create the ideal matrix based on linguistic rules (one time)
            self._ideal_compatibility_matrix = self._create_ideal_compatibility_matrix()
        # L2 loss between current matrix and ideal matrix
        linguistic_loss = F.mse_loss(
            self.compatibility_matrix, 
            self._ideal_compatibility_matrix.to(self.compatibility_matrix.device)
            )
            
        return strength * linguistic_loss
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