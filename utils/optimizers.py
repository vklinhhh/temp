# utils/optimizers.py
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)


def create_optimizer(model, learning_rate, weight_decay=0.01, discriminative_lr=False, encoder_lr_factor=0.1):
    """Create optimizer with special handling for compatibility matrix"""
    
    if discriminative_lr:
        compatibility_params = []
        encoder_params = []
        decoder_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'character_diacritic_compatibility.compatibility_matrix' in name:
                compatibility_params.append(param)
            elif name.startswith('vision_encoder.'):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        # 🔥 SOLUTION: Much lower LR for compatibility matrix + high weight decay
        optimizer = optim.AdamW([
            {
                'params': compatibility_params, 
                'lr': learning_rate * 0.01,  # 100x smaller LR
                'weight_decay': 0.1  # High weight decay to prevent drift
            },
            {'params': encoder_params, 'lr': learning_rate * encoder_lr_factor},
            {'params': decoder_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
    else:
        # Handle compatibility matrix separately even in non-discriminative mode
        compatibility_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'character_diacritic_compatibility.compatibility_matrix' in name:
                compatibility_params.append(param)
            else:
                other_params.append(param)
        
        if compatibility_params:
            optimizer = optim.AdamW([
                {
                    'params': compatibility_params,
                    'lr': learning_rate * 0.01,  # Much smaller LR
                    'weight_decay': 0.1
                },
                {'params': other_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(other_params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer