# model/dynamic_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class DynamicMultiScaleFusion(nn.Module):
    """
    Dynamic Multi-Scale Feature Fusion module.
    
    Instead of fixed weights for combining features from different encoder layers,
    this module learns to dynamically weight the contribution of each layer based on
    the input content, allowing the model to adapt its fusion strategy per image.
    """
    def __init__(self, encoder_output_size, num_layers=2, fusion_dim=None, use_gate=True):
        """
        Initialize the dynamic fusion module.
        
        Args:
            encoder_output_size: The dimension of each encoder layer output
            num_layers: Number of encoder layers to fuse
            fusion_dim: Output dimension after fusion (defaults to encoder_output_size)
            use_gate: Whether to use gating mechanism for input-adaptive fusion
        """
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.num_layers = num_layers
        self.fusion_dim = fusion_dim if fusion_dim is not None else encoder_output_size
        self.use_gate = use_gate
        
        # Global context encoder to generate fusion weights for each layer
        self.context_encoder = nn.Sequential(
            nn.Linear(encoder_output_size * num_layers, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_layers),  # One weight per layer
        )
        
        # Optional attention-based weighting (instead of global weights)
        if self.use_gate:
            # Create gate generators for each layer
            self.gate_generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(encoder_output_size, encoder_output_size // 2),
                    nn.LayerNorm(encoder_output_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(encoder_output_size // 2, 1),  # Scalar weight per position
                )
                for _ in range(num_layers)
            ])
        
        # Projection layer to map fused features to desired output dimension
        self.fusion_projection = nn.Sequential(
            nn.Linear(encoder_output_size, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features_list):
        """
        Apply dynamic fusion to a list of features from different encoder layers.
        
        Args:
            features_list: List of tensor features from different encoder layers
                          [layer_1_features, layer_2_features, ...]
                          Each tensor has shape [batch_size, seq_length, encoder_output_size]
        
        Returns:
            fused_features: Tensor of shape [batch_size, seq_length, fusion_dim]
        """
        batch_size, seq_length, _ = features_list[0].shape
        
        # Method 1: Global layer importance using sequence mean features
        # Concatenate mean features from each layer
        global_features = []
        for features in features_list:
            # Global pooling to get a representation of the entire sequence
            mean_features = torch.mean(features, dim=1)  # [batch_size, encoder_output_size]
            global_features.append(mean_features)
        
        # Concatenate global features from all layers
        # [batch_size, encoder_output_size * num_layers]
        concatenated_global = torch.cat(global_features, dim=-1)
        
        # Generate global fusion weights
        # [batch_size, num_layers]
        global_weights = self.context_encoder(concatenated_global)
        global_weights = F.softmax(global_weights, dim=-1)
        
        # Apply global weights to each layer
        globally_weighted = []
        for i, features in enumerate(features_list):
            # [batch_size, 1, 1] * [batch_size, seq_length, encoder_output_size]
            layer_weight = global_weights[:, i:i+1, None]
            weighted_features = features * layer_weight.expand_as(features)
            globally_weighted.append(weighted_features)
        
        # Method 2: Position-specific gating (if enabled)
        if self.use_gate:
            # Apply gates to each position in each layer
            gated_features = []
            for i, features in enumerate(features_list):
                # Generate gates: [batch_size, seq_length, 1]
                gates = torch.sigmoid(self.gate_generators[i](features))
                # Apply gates: [batch_size, seq_length, encoder_output_size]
                gated = features * gates
                gated_features.append(gated)
            
            # Combine global weights and gates
            # Sum the gated features
            fused_features = sum(gated_features)
        else:
            # Use only globally weighted features
            fused_features = sum(globally_weighted)
        
        # Apply final projection
        fused_features = self.fusion_projection(fused_features)
        
        return fused_features
    
    def extra_repr(self):
        """Return a string representation of module parameters."""
        return f"encoder_size={self.encoder_output_size}, num_layers={self.num_layers}, fusion_dim={self.fusion_dim}, use_gate={self.use_gate}"


class LocalFeatureEnhancer(nn.Module):
    """
    Local Feature Enhancement module specifically designed for diacritical marks.
    
    This module enhances local feature detection for subtle visual elements like
    diacritical marks by applying specialized convolution filters and attention
    mechanisms to focus on fine-grained details.
    """
    def __init__(self, feature_dim, num_diacritics=None, use_spatial_attention=True):
        """
        Initialize the local feature enhancer.
        
        Args:
            feature_dim: Dimension of input features
            num_diacritics: Number of diacritic types (for specialized attention)
            use_spatial_attention: Whether to use spatial attention mechanism
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_spatial_attention = use_spatial_attention
        self.num_diacritics = num_diacritics
        
        # Feature transformation for enhancing local patterns
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Fine-grained pattern detector (multiple attention heads)
        self.num_heads = 4
        self.head_dim = feature_dim // self.num_heads
        assert self.head_dim * self.num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # Multi-head attention for detecting fine-grained patterns
        self.pattern_query = nn.Linear(feature_dim, feature_dim)
        self.pattern_key = nn.Linear(feature_dim, feature_dim)
        self.pattern_value = nn.Linear(feature_dim, feature_dim)
        self.pattern_output = nn.Linear(feature_dim, feature_dim)
        
        # Spatial attention if enabled
        if use_spatial_attention:
            # Specialized detectors for regions where diacritics typically appear
            # (above, middle, below character regions)
            self.region_attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim // 2, 3)  # 3 regions: above, middle, below
            )
            
            # Region-specific feature enhancers
            self.region_enhancers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
                for _ in range(3)
            ])
        
        # Diacritic-specific feature enhancement if num_diacritics provided
        if num_diacritics and num_diacritics > 0:
            # Learnable prototypes for each diacritic type
            self.diacritic_prototypes = nn.Parameter(
                torch.randn(num_diacritics, feature_dim)
            )
            
            # Projection for diacritic detection
            self.diacritic_detector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, num_diacritics)
            )
            
            # Add prototype initialization from a specific method
            nn.init.xavier_uniform_(self.diacritic_prototypes)
        
        # Final layer normalization for stable training
        self.final_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """
        Enhance local features in the input.
        
        Args:
            features: Tensor of shape [batch_size, seq_length, feature_dim]
        
        Returns:
            enhanced_features: Tensor of shape [batch_size, seq_length, feature_dim]
        """
        batch_size, seq_length, _ = features.shape
        
        # 1. Apply feature transformation for local pattern enhancement
        transformed = self.feature_transform(features)
        
        # 2. Apply multi-head attention for fine-grained pattern detection
        # Compute query, key, value
        query = self.pattern_query(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.pattern_key(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.pattern_value(features).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        pattern_context = torch.matmul(attention_weights, value)
        pattern_context = pattern_context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        pattern_output = self.pattern_output(pattern_context)
        
        # Add residual connection
        enhanced = transformed + pattern_output
        
        # 3. Apply spatial attention if enabled
        if self.use_spatial_attention:
            # Generate region attention weights
            region_weights = F.softmax(self.region_attention(enhanced), dim=-1)  # [batch, seq_len, 3]
            
            # Apply region-specific enhancement
            region_enhanced = torch.zeros_like(enhanced)
            for i, enhancer in enumerate(self.region_enhancers):
                # Enhanced features for this region
                region_output = enhancer(enhanced)
                # Weight by region attention
                region_enhanced += region_weights[:, :, i:i+1] * region_output
            
            # Combine with original enhanced features
            enhanced = enhanced + region_enhanced
        
        # 4. Apply diacritic-specific enhancement if enabled
        if hasattr(self, 'diacritic_prototypes') and hasattr(self, 'diacritic_detector'):
            # Compute similarity to diacritic prototypes
            diacritic_logits = self.diacritic_detector(enhanced)  # [batch, seq_len, num_diacritics]
            diacritic_probs = F.softmax(diacritic_logits, dim=-1)
            
            # Weighted sum of prototypes
            prototype_features = torch.matmul(diacritic_probs, self.diacritic_prototypes)  # [batch, seq_len, feature_dim]
            
            # Add prototype features (with scaling to prevent domination)
            enhanced = enhanced + 0.2 * prototype_features
        
        # Final normalization
        enhanced = self.final_norm(enhanced)
        
        return enhanced
    
    def extra_repr(self):
        """Return a string representation of module parameters."""
        return f"feature_dim={self.feature_dim}, num_heads={self.num_heads}, use_spatial_attention={self.use_spatial_attention}"