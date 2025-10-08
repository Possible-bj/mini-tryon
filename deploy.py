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
    if not Path("requirements_standalone.txt").exists():
        print_error("requirements_standalone.txt not found!")
        print_error("Please run this script from the STEP_BY_STEP directory")
        sys.exit(1)
    
    # Step 1: Fix Pillow installation (common issue)
    if not run_command(
        [python_cmd, "-m", "pip", "uninstall", "-y", "pillow", "pil"],
        "Removing old Pillow installation"
    ):
        pass  # Continue even if uninstall fails
    
    if not run_command(
        [python_cmd, "-m", "pip", "install", "--no-cache-dir", "pillow"],
        "Installing fresh Pillow"
    ):
        sys.exit(1)
    
    # Step 2: Install blinker
    if not run_command(
        [python_cmd, "-m", "pip", "install", "--ignore-installed", "blinker"],
        "Installing blinker"
    ):
        sys.exit(1)
    
    # Step 3: Install requirements
    if not run_command(
        [python_cmd, "-m", "pip", "install", "-r", "requirements_standalone.txt"],
        "Installing dependencies from requirements_standalone.txt"
    ):
        sys.exit(1)
    
    # Step 4: Install Flask dependencies
    if not run_command(
        [python_cmd, "-m", "pip", "install", "flask", "flask-cors"],
        "Installing Flask dependencies"
    ):
        sys.exit(1)
    
    # Step 5: Download models
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
        sys.exit(1)
    
    # Success message
    print()
    print("=" * 50)
    print(f"{Colors.GREEN}Deployment completed successfully!{Colors.NC}")
    print("=" * 50)
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

