"""
Fine-tuning Qwen2.5-Omni-3B with Custom Wrapper

This approach creates a custom wrapper around the Omni model to expose
a forward() method compatible with the Trainer API, while preserving
the multimodal capabilities intact.

The key challenge with Qwen2.5-Omni is its unusual architecture:
- Qwen2_5OmniForConditionalGeneration contains a 'thinker' (language model)
  and 'talker' (speech synthesis), plus multimodal encoders
- Standard SFTTrainer doesn't work because forward() is not implemented
  on the main wrapper
- Solution: Apply LoRA to the Thinker BEFORE wrapping, then expose it
  through a custom forward() method

Usage:
    python finetune_omni_wrapper.py --dataset ../../data/dataset_v3.jsonl \
                                     --model-path /path/to/qwen25_omni \
                                     --output qwen_omni_finetuned_lora
"""

import json
import os
import argparse
from typing import List, Dict
import torch
import torch.nn as nn


def load_jsonl_dataset(filepath: str) -> List[Dict]:
    """
    Load training examples from JSONL file.

    Supports three types of examples:
    - chat: Simple user/assistant conversations
    - analysis: Context-aware analysis with metadata
    - analysis_json: Analysis with full JSON payload context

    Args:
        filepath: Path to JSONL dataset file

    Returns:
        List of examples with formatted messages
    """
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            messages = []

            if data['type'] == 'chat':
                # Simple chat format
                messages = [
                    {"role": "user", "content": data['messages'][0]['content']},
                    {"role": "assistant", "content": data['messages'][1]['content']}
                ]
            elif data['type'] == 'analysis':
                # Analysis with context metadata
                user_msg = data['messages'][0]['content']
                assistant_msg = data['messages'][1]['content']
                context = data.get('context', {})
                day_of_flower = context.get('today_log', {}).get('day_of_flower', 'N/A')
                messages = [
                    {"role": "user", "content": f"[Day flo: {day_of_flower}] {user_msg}"},
                    {"role": "assistant", "content": assistant_msg}
                ]
            elif data['type'] == 'analysis_json':
                # Analysis with full JSON context
                context_str = json.dumps(data['context'], ensure_ascii=False, indent=2)
                user_msg = data['messages'][0]['content']
                assistant_msg = data['messages'][1]['content']
                full_user_msg = f"{user_msg}\n\nJSON Payload:\n```json\n{context_str}\n```"
                messages = [
                    {"role": "user", "content": full_user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]

            examples.append({"messages": messages})

    print(f"✅ Loaded {len(examples)} examples from {filepath}")
    return examples


class Qwen2_5OmniWrapper(nn.Module):
    """
    Custom wrapper around Qwen2_5OmniForConditionalGeneration.

    Exposes a forward() method compatible with Trainer for text-only fine-tuning,
    while preserving the full multimodal architecture for inference.

    Key features:
    - Accesses the internal 'thinker' (language model component)
    - Implements forward() for training
    - Delegates generate() to the full Omni model for multimodal inference
    - Supports LoRA adapters when applied to the thinker before wrapping
    """

    def __init__(self, omni_model):
        super().__init__()
        self.omni_model = omni_model

        # Access the Thinker which contains the actual generative model
        if hasattr(omni_model, 'thinker'):
            self.thinker = omni_model.thinker
            print("✅ Wrapper: accessed Thinker (Qwen2_5OmniThinkerForConditionalGeneration)")
        else:
            raise AttributeError("Omni model has no 'thinker' attribute")

        self.config = omni_model.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass compatible with Trainer for text-only training.

        Calls the Thinker directly, removing any multimodal inputs
        to focus on text-only fine-tuning.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for loss computation
            **kwargs: Additional arguments (multimodal inputs filtered out)

        Returns:
            Model outputs with loss
        """
        # Remove multimodal keys if present
        kwargs_clean = {k: v for k, v in kwargs.items()
                       if k not in ['pixel_values', 'audio_values', 'video_values',
                                    'audio_feature_lengths', 'pixel_values_videos']}

        # Call the Thinker directly
        # The Thinker handles both text and multimodal, but we only give it text
        outputs = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs_clean
        )
        return outputs

    def generate(self, *args, **kwargs):
        """
        Delegate generation to the full Omni model.

        This ensures multimodal capabilities are preserved during inference.
        """
        return self.omni_model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate input preparation to the Omni model."""
        if hasattr(self.omni_model, 'prepare_inputs_for_generation'):
            return self.omni_model.prepare_inputs_for_generation(*args, **kwargs)
        # Fallback: return inputs as-is
        return args[0] if args else kwargs

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save LoRA adapters from the Thinker only.

        We only save adapters (not the full model) because the full Omni model
        has shared tensors that cause issues with standard serialization.
        """
        if hasattr(self.thinker, 'save_pretrained'):
            self.thinker.save_pretrained(save_directory, **kwargs)
        else:
            # Fallback: try to save the full model
            self.omni_model.save_pretrained(save_directory, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the Thinker."""
        if hasattr(self.thinker, 'gradient_checkpointing_enable'):
            self.thinker.gradient_checkpointing_enable(**kwargs)

    def parameters(self, recurse=True):
        """
        Return wrapper parameters.

        Important: After get_peft_model(), LoRA adapters are in the wrapper,
        not in omni_model, so we must return the wrapper's parameters.
        """
        return super().parameters(recurse=recurse)

    def named_parameters(self, *args, **kwargs):
        """Return named parameters of the wrapper (including LoRA after get_peft_model)."""
        return super().named_parameters(*args, **kwargs)


def finetune_omni_with_wrapper(
    dataset: List[Dict],
    model_path: str,
    output_dir: str,
    max_steps: int = 100,
    learning_rate: float = 2e-4
):
    """
    Fine-tune Qwen2.5-Omni with custom wrapper approach.

    Key steps:
    1. Load base Omni model with 4-bit quantization
    2. Extract the Thinker component
    3. Apply LoRA to the Thinker BEFORE wrapping (crucial for gradients)
    4. Replace the Thinker in the Omni model with the LoRA version
    5. Create wrapper around the modified Omni model
    6. Train with standard Trainer
    7. Save only the LoRA adapters

    Args:
        dataset: List of training examples
        model_path: Path to base Qwen2.5-Omni model
        output_dir: Directory to save LoRA adapters
        max_steps: Number of training steps
        learning_rate: Learning rate for AdamW optimizer
    """

    print("\n" + "="*60)
    print("🔧 FINE-TUNING QWEN2.5-OMNI WITH CUSTOM WRAPPER")
    print("="*60)

    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling
    )
    from peft import PeftModel
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print("✅ Imports OK\n")

    # Configure 4-bit quantization
    print("🔧 Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Double quantization for better memory
        bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    print("✅ 4-bit configuration OK\n")

    # Load Omni model
    print(f"📦 Loading Qwen2.5-Omni from {model_path}...")
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        enable_audio_output=False  # Disable audio synthesis during training
    )
    print("✅ Omni model loaded\n")
    print(f"\n🔧 Loading LoRA adapters from {LORA_ADAPTER_PATH}...")
    thinker = base_model.thinker
    thinker_with_lora = PeftModel.from_pretrained(thinker, LORA_ADAPTER_PATH)
    base_model.thinker = thinker_with_lora

    # Load processor and tokenizer
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("✅ Processor loaded\n")

    # Prepare Thinker for LoRA (BEFORE creating wrapper)
    print("🔧 Preparing for LoRA...")

    # Extract the Thinker
    thinker = omni_model.thinker
    print("✅ Thinker extracted\n")

    # Enable gradient checkpointing on the Thinker
    if hasattr(thinker, 'gradient_checkpointing_enable'):
        thinker.gradient_checkpointing_enable()

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=16,  # LoRA alpha (scaling factor)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # Target attention and FFN layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the Thinker directly
    # CRITICAL: This must be done BEFORE wrapping, otherwise gradients won't propagate
    thinker_with_lora = get_peft_model(thinker, lora_config)
    print("✅ LoRA applied to Thinker\n")

    # Display trainable parameters
    thinker_with_lora.print_trainable_parameters()
    print("")

    # Replace the Thinker in omni_model with the LoRA version
    omni_model.thinker = thinker_with_lora

    # Create wrapper AFTER applying LoRA
    print("🔧 Creating wrapper...")
    model = Qwen2_5OmniWrapper(omni_model)
    print("✅ Wrapper created with Thinker+LoRA\n")

    # Prepare dataset
    print("📊 Preparing dataset...")

    def format_chat_template(messages: List[Dict]) -> str:
        """Format messages using ChatML template."""
        text = ""
        for msg in messages:
            if msg['role'] == 'user':
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        return text

    formatted_texts = []
    for example in dataset:
        text = format_chat_template(example['messages'])
        formatted_texts.append({"text": text})

    hf_dataset = Dataset.from_list(formatted_texts)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"✅ Dataset tokenized: {len(tokenized_dataset)} examples\n")

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not masked language modeling
    )

    # Training configuration
    print("⚙️ Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 not available
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",  # Disable automatic checkpointing (we save manually)
        optim="paged_adamw_8bit",  # 8-bit Adam optimizer to save memory
        report_to="none"  # Disable logging to external services
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("✅ Trainer configured\n")
    print("🚀 Starting fine-tuning...\n")
    print(f"   - Steps: {max_steps}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Batch size: 1 (gradient accumulation: 4)")
    print(f"   - Output: {output_dir}\n")

    # Train
    trainer.train()

    print("\n✅ Fine-tuning completed!\n")

    # Save ONLY the LoRA adapters (official PEFT method)
    print(f"💾 Saving LoRA adapters to {output_dir}...")

    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)

    # OFFICIAL PEFT METHOD: Extract only LoRA weights
    # This function is used internally by save_pretrained()
    # but we call it manually to avoid issues with shared tensors
    lora_state_dict = get_peft_model_state_dict(thinker_with_lora)

    print(f"   - {len(lora_state_dict)} LoRA tensors extracted")

    # Save with safetensors (no shared tensors because we only save adapters)
    save_file(lora_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))

    # Save PEFT config
    if hasattr(thinker_with_lora, 'peft_config'):
        peft_config = thinker_with_lora.peft_config['default']
        peft_config.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print("✅ LoRA adapters saved!\n")

    # Verify saved files
    print(f"📁 Saved files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"   - {file}: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen2.5-Omni with custom wrapper')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to JSONL dataset')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to Qwen2.5-Omni model')
    parser.add_argument('--output', type=str, default='qwen_omni_finetuned_lora',
                        help='Output directory for LoRA adapters')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🌱 AI Trainer - Qwen2.5-Omni Fine-tuning")
    print("="*60)

    if not os.path.exists(args.dataset):
        print(f"\n❌ Dataset not found: {args.dataset}")
        return

    if not os.path.exists(args.model_path):
        print(f"\n❌ Model not found: {args.model_path}")
        return

    print(f"\n📂 Dataset: {args.dataset}")
    print(f"📂 Model: {args.model_path}")
    examples = load_jsonl_dataset(args.dataset)

    if len(examples) == 0:
        print("❌ No examples in dataset!")
        return

    finetune_omni_with_wrapper(
        dataset=examples,
        model_path=args.model_path,
        output_dir=args.output,
        max_steps=args.max_steps,
        learning_rate=args.lr
    )

    print("\n" + "="*60)
    print("✅ OMNI CAPABILITIES PRESERVED + FINE-TUNED")
    print("="*60)


if __name__ == '__main__':
    main()
