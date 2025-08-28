#!/bin/bash

# Step 2: Core Models Installation Script
echo "🚀 Installing packages for Step 2: Core Models"
echo "=============================================="

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install packages
echo "📦 Installing required packages..."
pip install -r requirements_step2.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 To test Step 2, run:"
echo "   python test_step2.py"
echo ""
echo "📝 Note: First run will download models (~2GB)"
