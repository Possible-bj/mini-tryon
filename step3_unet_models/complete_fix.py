#!/usr/bin/env python3
"""
Complete fix for all dependency conflicts
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
    print("🔧 Complete fix for all dependency conflicts...")
    print("=" * 60)
    
    # Step 1: Uninstall all conflicting packages
    print("📦 Step 1: Uninstalling all conflicting packages...")
    packages = ["transformers", "diffusers", "huggingface_hub", "accelerate"]
    for package in packages:
        success, stdout, stderr = run_command(f"pip uninstall {package} -y")
        if success:
            print(f"✅ Successfully uninstalled {package}")
        else:
            print(f"⚠️  Warning for {package}: {stderr}")
    
    # Step 2: Install compatible versions in correct order
    print("\n📦 Step 2: Installing compatible versions in correct order...")
    
    # Install huggingface_hub first (base dependency)
    print("Installing huggingface_hub...")
    success, stdout, stderr = run_command("pip install 'huggingface_hub>=0.16.0,<0.17.0'")
    if not success:
        print(f"❌ Failed to install huggingface_hub: {stderr}")
        return False
    print("✅ huggingface_hub installed")
    
    # Install transformers (depends on huggingface_hub)
    print("Installing transformers...")
    success, stdout, stderr = run_command("pip install 'transformers==4.35.0'")
    if not success:
        print(f"❌ Failed to install transformers: {stderr}")
        return False
    print("✅ transformers installed")
    
    # Install diffusers (depends on both)
    print("Installing diffusers...")
    success, stdout, stderr = run_command("pip install 'diffusers==0.21.4'")
    if not success:
        print(f"❌ Failed to install diffusers: {stderr}")
        return False
    print("✅ diffusers installed")
    
    # Install accelerate (optional but useful)
    print("Installing accelerate...")
    success, stdout, stderr = run_command("pip install 'accelerate>=0.20.0,<0.21.0'")
    if not success:
        print(f"⚠️  Warning: accelerate installation failed: {stderr}")
    else:
        print("✅ accelerate installed")
    
    print("\n🎉 Complete fix successful! All packages are now compatible.")
    print("Try running Step 3 again.")
    return True

if __name__ == "__main__":
    main()
