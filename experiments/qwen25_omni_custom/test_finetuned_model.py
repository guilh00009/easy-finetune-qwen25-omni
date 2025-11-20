"""
Test script for fine-tuned Qwen2.5-Omni model.

Loads the base model, applies LoRA adapters, and tests generation
to verify that fine-tuning worked correctly.

Usage:
    python test_finetuned_model.py
"""

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from peft import PeftModel

print("=" * 60)
print("TESTING FINE-TUNED QWEN2.5-OMNI MODEL")
print("=" * 60)

# Paths configuration
base_model_path = "/workspace/cannamini_model"
lora_adapter_path = "/workspace/models/qwen_omni_finetuned_lora_FINAL"

# Configure 4-bit quantization
print("\n🔧 Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# Load base model
print(f"\n📦 Loading base model from {base_model_path}...")
base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    enable_audio_output=False
)
print("✅ Base model loaded")

# Load LoRA adapters onto the Thinker
print(f"\n🔧 Loading LoRA adapters from {lora_adapter_path}...")
thinker = base_model.thinker
thinker_with_lora = PeftModel.from_pretrained(thinker, lora_adapter_path)
base_model.thinker = thinker_with_lora
print("✅ LoRA adapters loaded onto Thinker")

# Load processor
print(f"\n📦 Loading processor...")
processor = Qwen2_5OmniProcessor.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)
print("✅ Processor loaded")

# Test question
print("\n" + "=" * 60)
print("FINE-TUNING TEST")
print("=" * 60)

question = "What is my growing setup?"
print(f"\n❓ Question: {question}\n")

# Format with chat template
messages = [
    {"role": "user", "content": question}
]

# Create prompt in ChatML format
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

# Tokenize
inputs = processor.tokenizer(prompt, return_tensors="pt").to(base_model.device)

# Generate
print("🤖 Generating response...")
with torch.no_grad():
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=processor.tokenizer.eos_token_id
    )

# Decode
response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display response
print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

# Extract just the assistant part
if "<|im_start|>assistant" in response:
    response = response.split("<|im_start|>assistant")[-1].strip()
if "<|im_end|>" in response:
    response = response.split("<|im_end|>")[0].strip()

print(f"\n{response}\n")

# Verify if response contains expected information
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

# Check for keywords that should appear if fine-tuning worked
# (These are specific to the training dataset used)
success_keywords = ["120", "tent", "pot", "8"]
found = [kw for kw in success_keywords if kw.lower() in response.lower()]

print(f"\n✓ Keywords found: {', '.join(found) if found else 'None'}")

if len(found) >= 2:
    print("\n✅ Fine-tuning appears to have worked!")
    print("   The model learned information from the dataset.")
else:
    print("\n⚠️  The model doesn't clearly mention the setup.")
    print("   Possible that fine-tuning didn't have enough effect with 100 steps.")

print("\n" + "=" * 60)
