#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Check if nvidia-smi exists and GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "GPU detected! Installing GPU-enabled PyTorch and JAX..."
    
    # Try to detect CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    
    # Map CUDA version to PyTorch index URL and JAX wheel
    case $CUDA_VERSION in
        12.2) 
            TORCH_URL="https://download.pytorch.org/whl/cu122"
            JAX_EXTRA="[cuda12]"
            ;;
        12.1|12.0) 
            TORCH_URL="https://download.pytorch.org/whl/cu121"
            JAX_EXTRA="[cuda12]"
            ;;
        11.8|11.7|11.6)
            TORCH_URL="https://download.pytorch.org/whl/cu118"
            JAX_EXTRA="[cuda11]"
            ;;
        *)
            echo "Unknown or unsupported CUDA version ($CUDA_VERSION). Installing CPU-only PyTorch and JAX."
            TORCH_URL=""
            JAX_EXTRA=""
            ;;
    esac

    # Install PyTorch
    if [ -n "$TORCH_URL" ]; then
        pip install torch torchvision torchaudio --index-url $TORCH_URL
    else
        pip install torch torchvision torchaudio
    fi

    # Install JAX (with CUDA if known)
    if [ -n "$JAX_EXTRA" ]; then
        pip install -U "jax$JAX_EXTRA" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        pip install -U jax
    fi

else
    echo "No GPU detected. Installing CPU-only PyTorch and JAX..."
    pip install torch torchvision torchaudio
    pip install -U jax
fi

# Install the rest of the packages
pip install -r requirements.txt

echo "Installation complete!"
