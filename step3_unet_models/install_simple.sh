#!/bin/bash

# Simplified Step 3: UNet Models Installation Script
echo "🚀 Installing packages for Simplified Step 3: UNet Models"
echo "========================================================"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Check if conda is available (alternative package manager)
if command -v conda &> /dev/null; then
    echo "📦 Conda detected. Using conda for installation..."
    echo "Installing packages with conda..."
    conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -y -c conda-forge diffusers transformers einops accelerate safetensors
else
    echo "📦 Using pip for installation..."
    
    # Upgrade pip first
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install PyTorch (with CUDA support if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other packages
    echo "Installing other packages..."
    pip install -r requirements_simple.txt
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 To test the simplified Step 3, run:"
echo "   python test_simple_step3.py"
echo ""
echo "📝 What this simplified version provides:"
echo "   - Uses latest stable diffusers versions"
echo "   - No complex hacks or modifications"
echo "   - Fallback mechanisms if models fail to load"
echo "   - Clean, maintainable code"
echo ""
echo "🔧 If you encounter issues:"
echo "   1. Check the error messages above"
echo "   2. Try running: pip install --upgrade diffusers transformers"
echo "   3. Make sure you have enough disk space for model downloads"
