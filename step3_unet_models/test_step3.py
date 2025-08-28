"""
Test script for Step 3: UNet Models
Run this to verify all UNet models load correctly.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from unet_models import main

if __name__ == "__main__":
    print("🧪 Testing Step 3: UNet Models")
    print("=" * 40)
    
    try:
        main()
        print("\n🎉 All UNet models loaded successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
