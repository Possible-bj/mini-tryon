#!/usr/bin/env python3
"""
Script to download missing ONNX models for IDM-VTON
"""

import os
import requests
import zipfile
from pathlib import Path
import hashlib

def download_file(url, filename, expected_size=None, expected_hash=None):
    """Download a file with progress bar and verification"""
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\n✓ Downloaded {filename}")
        
        # Verify file size
        actual_size = os.path.getsize(filename)
        if expected_size and actual_size < expected_size:
            print(f"⚠ Warning: File size ({actual_size}) is smaller than expected ({expected_size})")
            return False
        
        # Verify hash if provided
        if expected_hash:
            with open(filename, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != expected_hash:
                print(f"⚠ Warning: Hash mismatch. Expected: {expected_hash}, Got: {file_hash}")
                return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def main():
    """Main download function"""
    print("=== IDM-VTON Model Downloader ===")
    
    # Create ckpt directories if they don't exist
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    
    (ckpt_dir / "humanparsing").mkdir(exist_ok=True)
    (ckpt_dir / "openpose" / "ckpts").mkdir(parents=True, exist_ok=True)
    
    # Model URLs and information
    models = {
        "ckpt/humanparsing/parsing_atr.onnx": {
            "url": "https://cnb.cool/chengxianggongre/IDM-VTON/-/lfs/04c7d1d070d0e0ae943d86b18cb5aaaea9e278d97462e9cfb270cbbe4cd977f4?name=parsing_atr.onnx",
            "expected_size": 254000000,  # 254.50MB minimum
            "description": "ATR Human Parsing Model",  
        },
        "ckpt/humanparsing/parsing_lip.onnx": {
            "url": "https://cnb.cool/chengxianggongre/IDM-VTON/-/lfs/8436e1dae96e2601c373d1ace29c8f0978b16357d9038c17a8ba756cca376dbc?name=parsing_lip.onnx",
            "expected_size": 254000000,  # 254.50MB minimum
            "description": "LIP Human Parsing Model"
        },
        "ckpt/openpose/ckpts/body_pose_model.pth": {
            "url": "https://huggingface.co/spaces/pngwn/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth",
            "expected_size": 209000000,  # 209MB minimum
            "description": "OpenPose Body Pose Model"
        }
        
    }
    
    # Alternative sources for the models
    alternative_sources = {
        "ckpt/humanparsing/parsing_atr.onnx": [
            "https://github.com/EngineeringResearch/IDM-VTON/releases/download/v1.0/parsing_atr.onnx",
            "https://huggingface.co/spaces/IDM-VTON/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx"
        ],
        "ckpt/humanparsing/parsing_lip.onnx": [
            "https://github.com/EngineeringResearch/IDM-VTON/releases/download/v1.0/parsing_lip.onnx",
            "https://huggingface.co/spaces/IDM-VTON/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"
        ],
        "ckpt/openpose/ckpts/body_pose_model.pth": [
            "https://github.com/EngineeringResearch/IDM-VTON/releases/download/v1.0/body_pose_model.pth",
            "https://huggingface.co/spaces/IDM-VTON/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"
        ]
    }
    
    # Check what's missing
    missing_models = []
    for model_path, info in models.items():
        if not os.path.exists(model_path) or os.path.getsize(model_path) < info["expected_size"]:
            missing_models.append(model_path)
    
    if not missing_models:
        print("✓ All models are already present!")
        return
    
    print(f"Missing models: {', '.join(missing_models)}")
    print("\nAttempting to download missing models...")
    
    # Try to download each missing model
    for model_path in missing_models:
        print(f"\n--- Downloading {model_path} ---")
        
        # Try alternative sources first
        downloaded = False
        for i, url in enumerate(alternative_sources[model_path]):
            print(f"Trying source {i+1}: {url}")
            if download_file(url, model_path, models[model_path]["expected_size"]):
                downloaded = True
                break
        
        if not downloaded:
            print(f"✗ Failed to download {model_path} from all sources")
            print("You may need to download this model manually from:")
            print("1. The original IDM-VTON repository")
            print("2. Hugging Face Hub")
            print("3. Or use the service with basic functionality")
    
    print("\n=== Download Summary ===")
    for model_path in models:
        if os.path.exists(model_path) and os.path.getsize(model_path) >= models[model_path]["expected_size"]:
            print(f"✓ {model_path} - Ready")
        else:
            print(f"✗ {model_path} - Missing or incomplete")
    
    print("\nTo test the service:")
    print("1. Run: python test_service.py")
    print("2. The service will work with available models")
    print("3. Missing models will use fallback methods")

if __name__ == "__main__":
    main()
