#!/usr/bin/env python3
"""
Script to download missing ONNX models for ChangeClothesAI
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print(f"\n✓ Downloaded {filename}")

def main():
    """Main download function"""
    print("=== ChangeClothesAI Model Downloader ===")
    
    # Create ckpt directories if they don't exist
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    
    (ckpt_dir / "humanparsing").mkdir(exist_ok=True)
    (ckpt_dir / "openpose" / "ckpts").mkdir(parents=True, exist_ok=True)
    
    # Check what's missing
    missing_models = []
    
    if not os.path.exists("ckpt/humanparsing/parsing_atr.onnx") or os.path.getsize("ckpt/humanparsing/parsing_atr.onnx") < 1000000:
        missing_models.append("parsing_atr.onnx")
    
    if not os.path.exists("ckpt/humanparsing/parsing_lip.onnx") or os.path.getsize("ckpt/humanparsing/parsing_lip.onnx") < 1000000:
        missing_models.append("parsing_lip.onnx")
    
    if not os.path.exists("ckpt/openpose/ckpts/body_pose_model.pth") or os.path.getsize("ckpt/openpose/ckpts/body_pose_model.pth") < 1000000:
        missing_models.append("body_pose_model.pth")
    
    if not missing_models:
        print("✓ All models are already present!")
        return
    
    print(f"Missing models: {', '.join(missing_models)}")
    print("\nNote: These models are large files that need to be downloaded separately.")
    print("You have a few options:")
    print("\n1. Use Git LFS (if this is a git repository):")
    print("   git lfs pull")
    print("   git lfs fetch --all")
    print("   git lfs checkout")
    
    print("\n2. Download from the original project repository")
    print("   (You'll need to find the original source)")
    
    print("\n3. Use the service without these models (will use basic masking)")
    print("   The service will still work but with reduced functionality.")
    
    print("\n4. Try to download from Hugging Face:")
    print("   The models might be available on Hugging Face Hub")
    
    # Try to download from some common sources
    print("\n=== Attempting to download from common sources ===")
    
    # You can add specific download URLs here if you find them
    # For now, we'll just provide instructions
    
    print("\nTo proceed without these models:")
    print("1. Run: python test_service.py")
    print("2. The service will use basic masking instead")
    print("3. Results may be less accurate but the service will work")
    
    print("\nTo get the full functionality:")
    print("1. Find the original project repository")
    print("2. Download the model files manually")
    print("3. Place them in the ckpt/ subdirectories")
    print("4. Restart the service")

if __name__ == "__main__":
    main()
