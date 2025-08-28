#!/bin/bash

# Quick fix for huggingface_hub compatibility
echo "🔧 Quick fix for huggingface_hub compatibility..."
echo "=============================================="

echo "📦 Uninstalling incompatible packages..."
pip uninstall huggingface_hub -y

echo "📦 Installing compatible huggingface_hub..."
pip install 'huggingface_hub>=0.16.0,<0.17.0'

echo "✅ Fix complete! Try running Step 3 again."
