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
    print("🔧 Fixing diffusers compatibility issues...")
    print("=" * 50)
    
    # Uninstall current diffusers
    print("📦 Uninstalling current diffusers...")
    success, stdout, stderr = run_command("pip uninstall diffusers -y")
    if success:
        print("✅ Successfully uninstalled diffusers")
    else:
        print(f"⚠️  Warning: {stderr}")
    
    # Install compatible version
    print("📦 Installing compatible diffusers version...")
    success, stdout, stderr = run_command("pip install 'diffusers>=0.24.0,<0.25.0'")
    if success:
        print("✅ Successfully installed compatible diffusers")
        print(stdout)
    else:
        print(f"❌ Failed to install diffusers: {stderr}")
        return False
    
    print("\n🎉 Fix complete! Try running Step 3 again.")
    return True

if __name__ == "__main__":
    main()
