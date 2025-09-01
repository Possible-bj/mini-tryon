#!/usr/bin/env python3
"""
Check and analyze current results
"""

import os
from PIL import Image

def check_results():
    """Check what results we currently have"""
    print("🔍 Checking Current Results")
    print("=" * 40)
    
    # List all result files
    result_files = [
        "hr_viton_result.png",
        "quick_tryon_result.png", 
        "quick_comparison.png",
        "stable_diffusion_tryon.png",
        "simple_blend_tryon.png",
        "comparison_fallback.png"
    ]
    
    print("📁 Available result files:")
    for file in result_files:
        if os.path.exists(file):
            try:
                img = Image.open(file)
                print(f"   ✅ {file}: {img.size} ({img.mode})")
            except Exception as e:
                print(f"   ❌ {file}: Error opening - {e}")
        else:
            print(f"   ⚠️  {file}: Not found")
    
    print(f"\n🎯 Current Status:")
    print(f"   - HR-VITON: Failed to download models (repository doesn't exist)")
    print(f"   - Fallback: Created simple models that generated output")
    print(f"   - Result: hr_viton_result.png was created")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Check hr_viton_result.png to see what quality we got")
    print(f"   2. Try real_tryon_model.py for better results")
    print(f"   3. Consider using a different pre-trained model")
    
    return True

if __name__ == "__main__":
    check_results()
