#!/usr/bin/env python3
"""
IDM-VTON Deployment Script
Cross-platform deployment automation
"""

import sys
import subprocess
import os
from pathlib import Path

# ANSI color codes
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    
    @staticmethod
    def disable_colors():
        """Disable colors for Windows CMD"""
        Colors.BLUE = ''
        Colors.GREEN = ''
        Colors.RED = ''
        Colors.NC = ''

# Disable colors on Windows if not in a terminal that supports ANSI
if sys.platform == 'win32' and not os.environ.get('ANSICON'):
    Colors.disable_colors()

def print_status(msg):
    print(f"{Colors.BLUE}[*]{Colors.NC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[✓]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[✗]{Colors.NC} {msg}")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print_status(description)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print_success(f"{description} - Done")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=" * 50)
    print("IDM-VTON Deployment Script")
    print("=" * 50)
    print()
    
    # Get Python executable - try multiple approaches
    python_cmd = sys.executable
    
    # Verify Python works by testing import
    try:
        import subprocess
        result = subprocess.run([python_cmd, "--version"], 
                              capture_output=True, text=True, check=True)
        print_success(f"Using Python: {python_cmd} ({result.stdout.strip()})")
    except Exception as e:
        print_error(f"Python verification failed: {e}")
        print_error("Please ensure Python is properly installed")
        sys.exit(1)
    
    print()
    
    # Check if we're in the right directory
    requirements_file = "requirements_minimal.txt"
    if not Path(requirements_file).exists():
        # Fallback to original requirements
        requirements_file = "requirements_standalone.txt"
        if not Path(requirements_file).exists():
            print_error("No requirements file found!")
            print_error("Please ensure requirements_minimal.txt or requirements_standalone.txt exists")
            sys.exit(1)
    
    print_success(f"Using requirements file: {requirements_file}")
    
    # Step 1: Upgrade pip and essential tools
    print_status("Upgrading pip and build tools...")
    if not run_command(
        [python_cmd, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading pip, setuptools, and wheel"
    ):
        print_error("Failed to upgrade pip - continuing anyway")
    
    # Step 2: Install PyTorch first (often needs specific handling)
    print_status("Installing PyTorch...")
    if not run_command(
        [python_cmd, "-m", "pip", "install", "torch>=2.0.0", "torchvision>=0.15.0"],
        "Installing PyTorch"
    ):
        print_error("PyTorch installation failed - trying CPU-only version")
        run_command(
            [python_cmd, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"],
            "Installing PyTorch CPU-only version"
        )
    
    # Step 3: Install core requirements
    print_status(f"Installing dependencies from {requirements_file}...")
    if not run_command(
        [python_cmd, "-m", "pip", "install", "-r", requirements_file],
        f"Installing dependencies from {requirements_file}"
    ):
        print_error(f"Failed to install from {requirements_file}")
        if requirements_file == "requirements_minimal.txt":
            print_status("Trying fallback to requirements_standalone.txt...")
            if Path("requirements_standalone.txt").exists():
                run_command(
                    [python_cmd, "-m", "pip", "install", "-r", "requirements_standalone.txt"],
                    "Installing from requirements_standalone.txt"
                )
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Step 4: Install additional packages that might be missing
    print_status("Installing additional packages...")
    additional_packages = [
        "config",
        "fvcore", 
        "pycocotools",
        "av",
        "blinker"
    ]
    
    for package in additional_packages:
        run_command(
            [python_cmd, "-m", "pip", "install", package],
            f"Installing {package}"
        )
    
    # Step 5: Try to install problematic packages with fallbacks
    print_status("Installing optional packages...")
    
    # Try BasicSR
    if not run_command(
        [python_cmd, "-m", "pip", "install", "basicsr>=1.4.2"],
        "Installing BasicSR"
    ):
        print_status("BasicSR pip install failed, trying from source...")
        run_command(
            [python_cmd, "-m", "pip", "install", "git+https://github.com/XPixelGroup/BasicSR.git"],
            "Installing BasicSR from source"
        )
    
    # Try Detectron2
    run_command(
        [python_cmd, "-m", "pip", "install", "detectron2", "-f", "https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"],
        "Installing Detectron2"
    )
    
    # Step 6: Verify critical imports
    print_status("Verifying critical imports...")
    critical_imports = [
        "torch",
        "torchvision", 
        "PIL",
        "cv2",
        "numpy",
        "transformers",
        "diffusers",
        "flask"
    ]
    
    failed_imports = []
    for module in critical_imports:
        try:
            result = subprocess.run(
                [python_cmd, "-c", f"import {module}; print('{module} - OK')"],
                capture_output=True, text=True, check=True
            )
            print_success(f"{module} - OK")
        except subprocess.CalledProcessError:
            print_error(f"{module} - FAILED")
            failed_imports.append(module)
    
    if failed_imports:
        print_error(f"Warning: {len(failed_imports)} critical imports failed:")
        for module in failed_imports:
            print_error(f"  - {module}")
        print_status("You may need to install these packages manually")
    
    # Step 7: Download models
    if not Path("download_models.py").exists():
        print_error("download_models.py not found!")
        sys.exit(1)
    
    print_status("Downloading models (this may take a while)...")
    try:
        result = subprocess.run(
            [python_cmd, "download_models.py"],
            check=True
        )
        print_success("Models downloaded")
    except subprocess.CalledProcessError as e:
        print_error("Failed to download models")
        print_status("You can download models manually later by running: python download_models.py")
        # Don't exit on model download failure - let user start service anyway
    
    # Success message
    print()
    print("=" * 50)
    if failed_imports:
        print(f"{Colors.GREEN}Deployment completed with warnings!{Colors.NC}")
        print(f"{Colors.RED}Some packages may need manual installation{Colors.NC}")
    else:
        print(f"{Colors.GREEN}Deployment completed successfully!{Colors.NC}")
    print("=" * 50)
    print()
    
    # Show status summary
    print("Installation Summary:")
    print("  ✓ Core dependencies installed")
    print("  ✓ PyTorch installed")
    print("  ✓ Additional packages installed")
    print("  ✓ Optional packages attempted")
    if failed_imports:
        print(f"  {Colors.RED}✗ {len(failed_imports)} critical imports failed{Colors.NC}")
    else:
        print("  ✓ All critical imports verified")
    
    print()
    print("Available commands:")
    print(f"  {python_cmd} api_service.py          # Start API service")
    print(f"  {python_cmd} download_models.py      # Download models manually")
    print(f"  {python_cmd} install_minimal.py      # Re-run minimal installation")
    print()
    
    # Ask if user wants to start the service
    try:
        response = input("Do you want to start the API service now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print_status("Starting API service...")
            subprocess.run([python_cmd, "api_service.py"])
        else:
            print()
            print("To start the service later, run:")
            print(f"  {python_cmd} api_service.py")
            print()
    except KeyboardInterrupt:
        print()
        print("Deployment complete. Service not started.")
        print()

if __name__ == "__main__":
    main()

