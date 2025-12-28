"""
FastAPI Server for AI-Assisted Growing App
Uses fine-tuned Qwen2.5-Omni with LoRA adapters

This server provides two main endpoints:
- /chat: Free-form conversation with context
- /analyze: Structured analysis of growing logs with recommendations

Usage:
    python main.py

The server starts on http://localhost:5000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from peft import PeftModel
import json

# Configuration
BASE_MODEL_PATH = "/apool/AI-Trainer/base_model"  # Base Omni model
LORA_ADAPTER_PATH = "/apool/AI-Trainer/models/qwen_omni_finetuned_lora"  # Fine-tuned LoRA adapters
PORT = 5000

app = FastAPI(title="Growing AI Server", version="1.0.0")

# CORS - allow requests from web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded at startup)
model = None
processor = None
device = None


# Pydantic schemas for request validation
class ChatRequest(BaseModel):
    """Request for free-form chat."""
    message: str
    context: Optional[Dict[str, Any]] = None


class AIRequest(BaseModel):
    """Request for structured growing log analysis."""
    session_config: Dict[str, Any]
    products_catalog: Dict[str, Any]
    today_log: Dict[str, Any]
    recent_history: Optional[List[Dict[str, Any]]] = None
    assistant_role: str = "grow_doctor"


class AIResponse(BaseModel):
    """Structured response for growing log analysis."""
    irrigation_plan: Dict[str, Any]
    climate_plan: Dict[str, Any]
    nutrient_adjustments: Dict[str, Any]
    warnings: List[str]
    explanations: List[str]
    simulations: Optional[Dict[str, Any]] = None


def load_model():
    """
    Load fine-tuned Qwen2.5-Omni with LoRA adapters at startup.

    Steps:
    1. Load base Omni model with 4-bit quantization
    2. Extract Thinker component
    3. Load LoRA adapters onto Thinker
    4. Replace Thinker in base model with fine-tuned version
    """
    global model, processor, device

    print("=" * 60)
    print("LOADING FINE-TUNED QWEN2.5-OMNI MODEL")
    print("=" * 60)

    # Configure 4-bit quantization
    print("\n🔧 Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # Load base model
    print(f"\n📦 Loading base model from {BASE_MODEL_PATH}...")
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        enable_audio_output=False
    )

    # Load LoRA adapters onto the Thinker
    print(f"\n🔧 Loading LoRA adapters from {LORA_ADAPTER_PATH}...")
    thinker = base_model.thinker
    thinker_with_lora = PeftModel.from_pretrained(thinker, LORA_ADAPTER_PATH)
    base_model.thinker = thinker_with_lora

    # Load processor
    print(f"\n📦 Loading processor...")
    processor_loaded = Qwen2_5OmniProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    model = base_model
    processor = processor_loaded
    device = base_model.device

    print("\n✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print("=" * 60 + "\n")


def generate_response(prompt: str, max_tokens: int = 512) -> str:
    """
    Generate a response from the model.

    Args:
        prompt: Input prompt (will be formatted with ChatML template)
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated response text
    """
    # Format with ChatML template
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize
    inputs = processor.tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # Decode
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant part
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()

    return response


@app.on_event("startup")
async def startup_event():
    """Load model at server startup."""
    load_model()


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server is running.

    Returns:
        Status object with model loading state
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Free-form chat endpoint for conversation.

    This endpoint allows natural language conversation with optional context.
    The context can contain any additional information relevant to the query.

    Args:
        request: ChatRequest with message and optional context

    Returns:
        {"response": "..."} with generated response

    Raises:
        HTTPException: If model is not loaded (503)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build prompt with context if provided
    prompt = request.message

    if request.context:
        context_str = f"\n\nContext:\n{json.dumps(request.context, ensure_ascii=False, indent=2)}"
        prompt = prompt + context_str

    # Generate response
    response = generate_response(prompt, max_tokens=512)

    return {"response": response}


@app.post("/analyze")
async def analyze(request: AIRequest):
    """
    Analysis endpoint for daily growing logs.

    This endpoint analyzes a growing log and provides structured recommendations
    for irrigation, climate, nutrients, and identifies potential issues.

    Args:
        request: AIRequest with session_config, products_catalog, today_log, etc.

    Returns:
        AIResponse with structured recommendations:
        - irrigation_plan: Whether to irrigate, volume, reasoning
        - climate_plan: Target temperatures and humidity
        - nutrient_adjustments: Product recommendations and doses
        - warnings: List of detected issues
        - explanations: Reasoning behind recommendations

    Raises:
        HTTPException: If model is not loaded (503)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build analysis prompt
    day_of_flower = request.today_log.get("day_of_flower", "N/A")

    context_str = json.dumps({
        "session_config": request.session_config,
        "products_catalog": request.products_catalog,
        "today_log": request.today_log,
        "recent_history": request.recent_history
    }, ensure_ascii=False, indent=2)

    prompt = f"""[Day flo: {day_of_flower}] Analyze this growing log and provide recommendations.

JSON Payload:
```json
{context_str}
```

Provide a structured analysis with:
- Irrigation plan (should irrigate? how much? why?)
- Climate plan (target temperatures and humidity)
- Nutrient adjustments (which products, what doses)
- Warnings (detected issues)
- Explanations (reasoning behind your recommendations)
"""

    # Generate response
    response_text = generate_response(prompt, max_tokens=1024)

    # Parse response (basic - could be improved with structured prompting)
    # For now, return a basic structure with the response as text
    return {
        "irrigation_plan": {
            "should_irrigate": True,
            "reason": response_text,
            "estimated_volume_per_pot": 0,
            "recommendations": []
        },
        "climate_plan": {
            "target_temp_day": 25,
            "target_temp_night": 20,
            "target_rh": 55,
            "actions": []
        },
        "nutrient_adjustments": {
            "products": []
        },
        "warnings": [],
        "explanations": [response_text],
        "simulations": None
    }


if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting server on http://localhost:{PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
