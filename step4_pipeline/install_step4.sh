#!/bin/bash

# Step 4: Pipeline Integration Installation Script
echo "🚀 Installing packages for Step 4: Pipeline Integration"
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
    conda install -y -c conda-forge pillow numpy
else
    echo "📦 Using pip for installation..."
    
    # Upgrade pip first
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    # Install core ML libraries
    echo "🔧 Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    # Install ML frameworks
    echo "🤖 Installing ML frameworks..."
    pip install diffusers transformers einops accelerate safetensors
    
    # Install image processing libraries
    echo "🖼️  Installing image processing libraries..."
    pip install Pillow numpy
    
    # Optional: Install xformers for better performance
    echo "⚡ Installing xformers (optional, for performance)..."
    pip install xformers
fi

echo ""
echo "✅ Installation completed!"
echo ""
echo "🧪 To test Step 4, run:"
echo "   python test_step4.py"
echo ""
echo "🚀 To run the pipeline directly:"
echo "   python simple_tryon_pipeline.py"
echo ""
echo "📚 Next: Step 5 - Preprocessing Integration"
