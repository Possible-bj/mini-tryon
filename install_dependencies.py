#!/usr/bin/env python3
"""
Robust dependency installation script for IDM-VTON
Handles problematic packages and provides fallbacks
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, allow_failure=False):
    """Run a command and handle errors"""
    print(f"[*] {description}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"[✓] {description} - Done")
        return True
    except subprocess.CalledProcessError as e:
        if allow_failure:
            print(f"[!] {description} - Failed (continuing)")
            print(f"Error: {e.stderr}")
            return False
        else:
            print(f"[✗] {description} - Failed")
            print(f"Error: {e.stderr}")
            return False

def install_package(package, description, allow_failure=False):
    """Install a single package"""
    return run_command([sys.executable, "-m", "pip", "install", package], 
                      description, allow_failure)

def main():
    print("🚀 Starting robust dependency installation for IDM-VTON")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements_standalone.txt").exists():
        print("[✗] requirements_standalone.txt not found!")
        print("[✗] Please run this script from the STEP_BY_STEP directory")
        sys.exit(1)
    
    # Step 1: Upgrade pip and setuptools
    print("\n📦 Upgrading pip and setuptools...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                "Upgrading pip, setuptools, and wheel")
    
    # Step 2: Install PyTorch first (often needs specific handling)
    print("\n🔥 Installing PyTorch...")
    if not install_package("torch>=2.0.0 torchvision>=0.15.0", "Installing PyTorch"):
        print("[!] PyTorch installation failed, trying CPU-only version...")
        install_package("torch torchvision --index-url https://download.pytorch.org/whl/cpu", 
                       "Installing PyTorch CPU-only", allow_failure=True)
    
    # Step 3: Install core dependencies from requirements file
    print("\n📋 Installing core dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_standalone.txt"],
                "Installing dependencies from requirements_standalone.txt")
    
    # Step 4: Handle problematic packages individually
    print("\n🔧 Installing problematic packages with fallbacks...")
    
    # Try BasicSR with different approaches
    print("\n📸 Installing BasicSR...")
    if not install_package("basicsr>=1.4.2", "Installing BasicSR", allow_failure=True):
        print("[!] BasicSR pip install failed, trying from source...")
        try:
            # Try installing from source
            run_command([sys.executable, "-m", "pip", "install", "git+https://github.com/XPixelGroup/BasicSR.git"],
                       "Installing BasicSR from source", allow_failure=True)
        except:
            print("[!] BasicSR installation completely failed - some features may not work")
    
    # Install Detectron2 (often needed for this project)
    print("\n🎯 Installing Detectron2...")
    install_package("detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html",
                   "Installing Detectron2", allow_failure=True)
    
    # Step 5: Install any missing dependencies that might be needed
    print("\n🔍 Installing additional dependencies...")
    additional_packages = [
        "opencv-contrib-python",
        "imageio",
        "imageio-ffmpeg",
        "scikit-learn",
        "seaborn",
        "plotly"
    ]
    
    for package in additional_packages:
        install_package(package, f"Installing {package}", allow_failure=True)
    
    # Step 6: Verify critical imports
    print("\n✅ Verifying critical imports...")
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
            __import__(module)
            print(f"[✓] {module} - OK")
        except ImportError as e:
            print(f"[✗] {module} - FAILED: {e}")
            failed_imports.append(module)
    
    # Final status
    print("\n" + "=" * 60)
    if failed_imports:
        print(f"⚠️  Installation completed with {len(failed_imports)} failed imports:")
        for module in failed_imports:
            print(f"   - {module}")
        print("\nYou may need to install these packages manually:")
        for module in failed_imports:
            print(f"   pip install {module}")
    else:
        print("🎉 All critical dependencies installed successfully!")
    
    print("\n🚀 Ready to run the IDM-VTON service!")
    print("Run: python api_service.py")

if __name__ == "__main__":
    main()
