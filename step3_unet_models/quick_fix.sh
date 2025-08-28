#!/bin/bash

# Quick fix for transformers, diffusers and huggingface_hub compatibility
echo "🔧 Quick fix for transformers, diffusers and huggingface_hub compatibility..."
echo "======================================================================"

echo "📦 Uninstalling incompatible packages..."
pip uninstall transformers diffusers huggingface_hub -y

echo "📦 Installing compatible versions..."
pip install 'transformers==4.35.0' 'diffusers>=0.24.0,<0.25.0' 'huggingface_hub>=0.16.0,<0.17.0'

echo "✅ Fix complete! Try running Step 3 again."
