# tests/test_compatibility_matrix.py
import torch

def test_compatibility_matrix_behavior(model):
    """Test that compatibility matrix enforces Vietnamese linguistic rules"""
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    dummy_features = torch.randn(batch_size, seq_len, model.config.shared_hidden_size)
    
    # Get base predictions for 'a' and 'b'
    base_logits = model.base_classifier(dummy_features)
    
    # Force base prediction to be 'a' vs 'b'
    a_idx = model.base_char_vocab.index('a')
    b_idx = model.base_char_vocab.index('b')
    
    # Create one-hot base predictions
    base_logits_a = torch.zeros_like(base_logits)
    base_logits_a[:, :, a_idx] = 10.0  # Force 'a' prediction
    
    base_logits_b = torch.zeros_like(base_logits)
    base_logits_b[:, :, b_idx] = 10.0  # Force 'b' prediction
    
    # Get compatibility effects
    if hasattr(model, 'character_diacritic_compatibility'):
        compat_bias_a, _ = model.character_diacritic_compatibility(base_logits_a, dummy_features)
        compat_bias_b, _ = model.character_diacritic_compatibility(base_logits_b, dummy_features)
        
        # Check expectations
        acute_idx = model.diacritic_vocab.index('acute')
        no_diac_idx = model.diacritic_vocab.index('no_diacritic')
        
        print(f"For base 'a':")
        print(f"  - Acute bias: {compat_bias_a[0, 0, acute_idx].item():.3f} (should be positive)")
        print(f"  - No diacritic bias: {compat_bias_a[0, 0, no_diac_idx].item():.3f}")
        
        print(f"For base 'b':")
        print(f"  - Acute bias: {compat_bias_b[0, 0, acute_idx].item():.3f} (should be negative)")
        print(f"  - No diacritic bias: {compat_bias_b[0, 0, no_diac_idx].item():.3f} (should be positive)")
        
        # Assertions
        assert compat_bias_a[0, 0, acute_idx] > 0, "Vowel 'a' should allow acute accent"
        assert compat_bias_b[0, 0, acute_idx] < 0, "Consonant 'b' should reject acute accent"
        assert compat_bias_b[0, 0, no_diac_idx] > compat_bias_b[0, 0, acute_idx], "'b' should prefer no_diacritic over acute"

# Run this test periodically during training
