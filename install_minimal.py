#!/usr/bin/env python3
"""
Minimal installation script for IDM-VTON
Uses only essential packages to avoid compilation issues
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
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
        print(f"[✗] {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Starting minimal dependency installation for IDM-VTON")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements_minimal.txt").exists():
        print("[✗] requirements_minimal.txt not found!")
        sys.exit(1)
    
    # Step 1: Upgrade pip
    print("\n📦 Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                "Upgrading pip")
    
    # Step 2: Install minimal requirements
    print("\n📋 Installing minimal dependencies...")
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"],
                       "Installing minimal dependencies"):
        print("[✗] Installation failed!")
        sys.exit(1)
    
    # Step 3: Install additional packages that might be needed
    print("\n🔧 Installing additional packages...")
    additional = [
        "config",
        "fvcore", 
        "pycocotools",
        "av"
    ]
    
    for package in additional:
        run_command([sys.executable, "-m", "pip", "install", package],
                   f"Installing {package}")
    
    # Step 4: Verify critical imports
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
    else:
        print("🎉 All critical dependencies installed successfully!")
    
    print("\n🚀 Ready to run the IDM-VTON service!")
    print("Run: python api_service.py")

if __name__ == "__main__":
    main()
