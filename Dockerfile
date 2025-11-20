FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /workspace

# Installer Python et build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Installer PyTorch 2.6+ avec CUDA 12.4
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Installer transformers version spéciale pour Qwen2.5-Omni
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356

# Installer dépendances de fine-tuning
RUN pip install --no-cache-dir \
    "datasets>=2.16.0" \
    "accelerate>=0.25.0" \
    "peft>=0.7.0" \
    "trl>=0.7.0" \
    "bitsandbytes>=0.41.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    scipy

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
