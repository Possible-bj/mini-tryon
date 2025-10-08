#!/bin/bash

# IDM-VTON Deployment Script
# This script automates the deployment process

set -e  # Exit on error

echo "================================"
echo "IDM-VTON Deployment Script"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Python is available
print_status "Checking Python installation..."

# Try common Python locations
PYTHON_CMD=""
for cmd in python3 python /usr/bin/python3 /usr/bin/python; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python is not installed or not in PATH"
    print_error "Tried: python3, python, /usr/bin/python3, /usr/bin/python"
    print_error "Please install Python or add it to your PATH"
    exit 1
fi

print_success "Python found: $PYTHON_CMD"

# Step 1: Install blinker
print_status "Installing blinker..."
$PYTHON_CMD -m pip install --ignore-installed blinker
print_success "Blinker installed"

# Step 2: Install requirements
print_status "Installing dependencies from requirements_standalone.txt..."
if [ ! -f "requirements_standalone.txt" ]; then
    print_error "requirements_standalone.txt not found!"
    exit 1
fi
$PYTHON_CMD -m pip install -r requirements_standalone.txt
print_success "Dependencies installed"

# Step 3: Install Flask dependencies for API
print_status "Installing Flask dependencies..."
$PYTHON_CMD -m pip install flask flask-cors
print_success "Flask dependencies installed"

# Step 4: Download models
print_status "Downloading models (this may take a while)..."
if [ ! -f "download_models.py" ]; then
    print_error "download_models.py not found!"
    exit 1
fi
$PYTHON_CMD download_models.py
print_success "Models downloaded"

echo ""
echo "================================"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "================================"
echo ""

# Ask if user wants to start the service
read -p "Do you want to start the API service now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting API service..."
    $PYTHON_CMD api_service.py
else
    echo ""
    echo "To start the service later, run:"
    echo "  $PYTHON_CMD api_service.py"
    echo ""
fi

