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

# Determine requirements file to use
REQUIREMENTS_FILE="requirements_minimal.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    REQUIREMENTS_FILE="requirements_standalone.txt"
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "No requirements file found!"
        print_error "Please ensure requirements_minimal.txt or requirements_standalone.txt exists"
        exit 1
    fi
fi

print_success "Using requirements file: $REQUIREMENTS_FILE"

# Step 1: Upgrade pip and essential tools
print_status "Upgrading pip and build tools..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel
print_success "Build tools upgraded"

# Step 2: Install PyTorch first (often needs specific handling)
print_status "Installing PyTorch..."
if ! $PYTHON_CMD -m pip install "torch>=2.0.0" "torchvision>=0.15.0"; then
    print_error "PyTorch installation failed - trying CPU-only version"
    $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch CPU-only version installed"
else
    print_success "PyTorch installed"
fi

# Step 3: Install core requirements
print_status "Installing dependencies from $REQUIREMENTS_FILE..."
if ! $PYTHON_CMD -m pip install -r "$REQUIREMENTS_FILE"; then
    print_error "Failed to install from $REQUIREMENTS_FILE"
    if [ "$REQUIREMENTS_FILE" = "requirements_minimal.txt" ] && [ -f "requirements_standalone.txt" ]; then
        print_status "Trying fallback to requirements_standalone.txt..."
        $PYTHON_CMD -m pip install -r requirements_standalone.txt
    else
        exit 1
    fi
fi
print_success "Core dependencies installed"

# Step 4: Install additional packages that might be missing
print_status "Installing additional packages..."
additional_packages=("config" "fvcore" "pycocotools" "av" "blinker")
for package in "${additional_packages[@]}"; do
    $PYTHON_CMD -m pip install "$package"
    print_success "$package installed"
done

# Step 5: Try to install problematic packages with fallbacks
print_status "Installing optional packages..."

# Try BasicSR
if ! $PYTHON_CMD -m pip install "basicsr>=1.4.2"; then
    print_status "BasicSR pip install failed, trying from source..."
    $PYTHON_CMD -m pip install git+https://github.com/XPixelGroup/BasicSR.git
fi
print_success "BasicSR installation attempted"

# Try Detectron2
$PYTHON_CMD -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
print_success "Detectron2 installation attempted"

# Step 6: Verify critical imports
print_status "Verifying critical imports..."
critical_imports=("torch" "torchvision" "PIL" "cv2" "numpy" "transformers" "diffusers" "flask")
failed_imports=()

for module in "${critical_imports[@]}"; do
    if $PYTHON_CMD -c "import $module" 2>/dev/null; then
        print_success "$module - OK"
    else
        print_error "$module - FAILED"
        failed_imports+=("$module")
    fi
done

if [ ${#failed_imports[@]} -gt 0 ]; then
    print_error "Warning: ${#failed_imports[@]} critical imports failed:"
    for module in "${failed_imports[@]}"; do
        print_error "  - $module"
    done
    print_status "You may need to install these packages manually"
fi

# Step 7: Download models
print_status "Downloading models (this may take a while)..."
if [ ! -f "download_models.py" ]; then
    print_error "download_models.py not found!"
    exit 1
fi

if $PYTHON_CMD download_models.py; then
    print_success "Models downloaded"
else
    print_error "Failed to download models"
    print_status "You can download models manually later by running: $PYTHON_CMD download_models.py"
    # Don't exit on model download failure - let user start service anyway
fi

echo ""
echo "================================"
if [ ${#failed_imports[@]} -gt 0 ]; then
    echo -e "${GREEN}Deployment completed with warnings!${NC}"
    echo -e "${RED}Some packages may need manual installation${NC}"
else
    echo -e "${GREEN}Deployment completed successfully!${NC}"
fi
echo "================================"
echo ""

# Show status summary
echo "Installation Summary:"
echo "  ✓ Core dependencies installed"
echo "  ✓ PyTorch installed"
echo "  ✓ Additional packages installed"
echo "  ✓ Optional packages attempted"
if [ ${#failed_imports[@]} -gt 0 ]; then
    echo -e "  ${RED}✗ ${#failed_imports[@]} critical imports failed${NC}"
else
    echo "  ✓ All critical imports verified"
fi

echo ""
echo "Available commands:"
echo "  $PYTHON_CMD api_service.py          # Start API service"
echo "  $PYTHON_CMD download_models.py      # Download models manually"
echo "  $PYTHON_CMD install_minimal.py      # Re-run minimal installation"
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

