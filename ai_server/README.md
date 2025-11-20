# AI Server - FastAPI Backend for Growing App

FastAPI server that exposes the fine-tuned Qwen2.5-Omni model for the Growing AI application.

## Installation

### Option 1: Python Virtual Environment

```bash
cd ai_server
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Option 2: Use Docker Environment (Recommended)

The server can use the same Docker environment as fine-tuning:

```bash
cd ..  # Back to AITrainer root
docker-compose run --rm -p 5000:5000 ai-trainer python /workspace/ai_server/main.py
```

## Starting the Server

### Python Direct Mode (if venv activated)

```bash
python main.py
```

### Docker Mode

```bash
cd ..  # Back to AITrainer root
docker-compose run --rm -p 5000:5000 ai-trainer python /workspace/ai_server/main.py
```

The server starts on **http://localhost:5000**

## Endpoints

### GET /health
Server health check

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /chat
Free-form conversation with the model

**Request:**
```json
{
  "message": "What is my growing setup?",
  "context": {
    "session_config": { ... },
    "products_catalog": { ... }
  }
}
```

**Response:**
```json
{
  "response": "Your setup includes a 120x120x200cm tent with 8 pots..."
}
```

### POST /analyze
Daily log analysis with structured recommendations

**Request:**
```json
{
  "session_config": { ... },
  "products_catalog": { ... },
  "today_log": {
    "day_of_flower": 28,
    "temp_day": 25,
    "temp_night": 20,
    ...
  },
  "recent_history": [ ... ],
  "assistant_role": "grow_doctor"
}
```

**Response:**
```json
{
  "irrigation_plan": {
    "should_irrigate": true,
    "reason": "...",
    "estimated_volume_per_pot": 2.5,
    "recommendations": [ ... ]
  },
  "climate_plan": { ... },
  "nutrient_adjustments": { ... },
  "warnings": [ ... ],
  "explanations": [ ... ]
}
```

## Configuration

### Model Paths

In `main.py`:

```python
BASE_MODEL_PATH = "../../CannaModMini/volume"  # Base Omni model
LORA_ADAPTER_PATH = "../models/qwen_omni_finetuned_lora_FINAL"  # LoRA adapters
PORT = 5000
```

**Note:** Update these paths to match your local setup.

### VRAM Required

- ~7.5 GB (4-bit quantization)
- NVIDIA GPU with CUDA support

## Connecting with Web App

1. Start the AI server (port 5000)
2. Open the Growing AI app
3. Go to **Settings → Application**
4. Verify the server URL is `http://localhost:5000`
5. Click "Test Connection" → should display "✓ Connected"

## Logs

The server displays:
- Model loading at startup
- Requests received and processed
- Any errors

## Troubleshooting

### Server won't start
- Check that port 5000 is not already in use
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### "Model not loaded"
- Verify that BASE_MODEL_PATH and LORA_ADAPTER_PATH are correct
- Check that the files exist

### CORS errors from the app
- Verify the server is running
- Check browser console for the exact URL being called

### Slow responses
- Normal: generation takes ~5-10 seconds in 4-bit
- With a more powerful GPU, it will be faster

## Code Structure

```
ai_server/
├── main.py              # FastAPI server
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## Next Steps

To improve the server:

1. **Structured response parsing** - use a prompt to generate JSON
2. **Response caching** - avoid regenerating the same analyses
3. **Streaming** - return response progressively
4. **Advanced logging** - save requests/responses
5. **Auth** - add authentication token
