#!/usr/bin/env python3
"""
Python environment check script
Helps diagnose Python installation issues
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python():
    print("=" * 60)
    print("Python Environment Check")
    print("=" * 60)
    
    # Check current Python
    print(f"Current Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 entries
    print()
    
    # Check PATH
    print("PATH environment variable:")
    path_dirs = os.environ.get('PATH', '').split(':')
    python_dirs = [d for d in path_dirs if 'python' in d.lower() or d in ['/usr/bin', '/usr/local/bin']]
    for d in python_dirs[:5]:  # Show first 5 relevant directories
        print(f"  {d}")
    print()
    
    # Check for Python executables
    print("Looking for Python executables:")
    python_candidates = [
        'python', 'python3', 'python3.8', 'python3.9', 'python3.10', 'python3.11', 'python3.12',
        '/usr/bin/python', '/usr/bin/python3',
        '/usr/local/bin/python', '/usr/local/bin/python3'
    ]
    
    found_pythons = []
    for cmd in python_candidates:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                found_pythons.append((cmd, version))
                print(f"  ✓ {cmd}: {version}")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
    
    if not found_pythons:
        print("  ✗ No Python executables found!")
        return False
    
    print()
    
    # Check pip
    print("Checking pip:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✓ pip: {result.stdout.strip()}")
        else:
            print(f"  ✗ pip failed: {result.stderr}")
    except Exception as e:
        print(f"  ✗ pip error: {e}")
    
    print()
    
    # Check current directory
    print("Current directory check:")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Script location: {Path(__file__).parent}")
    
    required_files = ['requirements_standalone.txt', 'download_models.py', 'api_service.py']
    print("  Required files:")
    for file in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {file}")
    
    print()
    print("=" * 60)
    
    if found_pythons:
        print("✅ Python environment looks good!")
        print(f"Recommended command: {found_pythons[0][0]} deploy.py")
        return True
    else:
        print("❌ Python environment has issues")
        print("Please install Python or fix PATH issues")
        return False

if __name__ == "__main__":
    check_python()
