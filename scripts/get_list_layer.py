import sys
import os
# Ensure your project root is in sys.path if running this snippet from a different directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust '..' if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel # Your model class
import json

# --- Paths (Update these to your actual paths) ---
MODEL_PATH = "p/root/hwt/VNHWT-Hierachical-Transformer-CTC/outputs/dynamic_fusion_large/best_model_hf"
COMBINED_VOCAB_PATH = "/root/hwt/VNHWT-Hierachical-Transformer-CTC/outputs/dynamic_fusion_large/combined_char_vocab.json"

# Load combined_char_vocab (needed for from_pretrained if not fully in config)
try:
    with open(COMBINED_VOCAB_PATH, 'r', encoding='utf-8') as f:
        combined_char_vocab_list = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Vocab file not found at {COMBINED_VOCAB_PATH}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {COMBINED_VOCAB_PATH}")
    sys.exit(1)

# Load the model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
        MODEL_PATH,
        combined_char_vocab=combined_char_vocab_list # Pass it if your config might be minimal
    )
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)


# --- Print all named modules ---
print("\n--- Named Modules in the Model ---")
for name, module in model.named_modules():
    # Optionally filter to show only certain types or skip very deep ones
    # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, TransformerEncoderLayer)):
    print(f"Layer Name: '{name}' \t Module Type: {type(module)}")

print("\n--- End of Named Modules ---")