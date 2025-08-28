#!/bin/bash

# Step 3: UNet Models Installation Script
echo "🚀 Installing packages for Step 3: UNet Models"
echo "=============================================="

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install packages
echo "📦 Installing required packages..."
pip install -r requirements_step3.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 To test Step 3, run:"
echo "   python test_step3.py"
echo ""
echo "📝 Note: This step requires:"
echo "   - einops for tensor operations"
echo "   - accelerate for model loading"
echo "   - safetensors for model weights"
echo "   - xformers (optional) for attention optimization"
