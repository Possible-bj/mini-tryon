#!/usr/bin/env python3
"""
Fix script for diffusers compatibility issues
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔧 Fixing transformers, diffusers and huggingface_hub compatibility issues...")
    print("=" * 70)
    
    # Uninstall current packages
    print("📦 Uninstalling current packages...")
    packages = ["transformers", "diffusers", "huggingface_hub"]
    for package in packages:
        success, stdout, stderr = run_command(f"pip uninstall {package} -y")
        if success:
            print(f"✅ Successfully uninstalled {package}")
        else:
            print(f"⚠️  Warning for {package}: {stderr}")
    
    # Install compatible versions
    print("📦 Installing compatible versions...")
    install_commands = [
        "pip install 'transformers==4.35.0'",
        "pip install 'diffusers>=0.24.0,<0.25.0'",
        "pip install 'huggingface_hub>=0.16.0,<0.17.0'"
    ]
    
    for cmd in install_commands:
        print(f"Running: {cmd}")
        success, stdout, stderr = run_command(cmd)
        if success:
            print("✅ Successfully installed")
        else:
            print(f"❌ Failed: {stderr}")
            return False
    
    print("\n🎉 Fix complete! Try running Step 3 again.")
    return True

if __name__ == "__main__":
    main()
