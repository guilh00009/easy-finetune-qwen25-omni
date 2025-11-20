"""
Inspection script to understand Qwen2.5-Omni architecture.

This utility script loads the Qwen2.5-Omni model and displays its internal
structure, helping to understand how to access the Thinker component and
which methods are available for fine-tuning.

Usage:
    python inspect_model.py
"""

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig

print("=" * 60)
print("QWEN2.5-OMNI ARCHITECTURE INSPECTION")
print("=" * 60)

# Configure 4-bit quantization to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

print("\n📦 Loading model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "/workspace/cannamini_model",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    enable_audio_output=False
)

print("✅ Model loaded\n")

print("=" * 60)
print("MODEL STRUCTURE")
print("=" * 60)

# Display main model attributes
print("\n📋 Main model attributes:")
for attr in dir(model):
    if not attr.startswith('_') and not callable(getattr(model, attr, None)):
        try:
            value = getattr(model, attr)
            if isinstance(value, torch.nn.Module):
                print(f"  - {attr}: {type(value).__name__}")
        except:
            pass

# Display submodules
print("\n🔧 Submodules (named_children):")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")
    # Display first-level sub-submodules
    for sub_name, sub_module in module.named_children():
        print(f"      - {sub_name}: {type(sub_module).__name__}")

# Check important methods
print("\n🔍 Important methods:")
methods_to_check = ['forward', '__call__', 'generate', 'prepare_inputs_for_generation']
for method in methods_to_check:
    has_method = hasattr(model, method) and callable(getattr(model, method, None))
    status = "✅" if has_method else "❌"
    print(f"  {status} {method}")

# Display detailed architecture
print("\n🏗️  Detailed architecture:")
print(model)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)
