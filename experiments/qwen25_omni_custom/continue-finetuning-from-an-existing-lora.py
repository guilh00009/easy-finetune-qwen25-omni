print(f"\n🔧 Loading LoRA adapters from {LORA_ADAPTER_PATH}...")
    thinker = base_model.thinker
    thinker_with_lora = PeftModel.from_pretrained(thinker, LORA_ADAPTER_PATH)
    base_model.thinker = thinker_with_lora
