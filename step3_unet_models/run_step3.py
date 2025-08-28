"""
Run Step 3 from the project root directory
This script should be run from the main IDM-VTON directory
"""

import os
import sys

# Change to the step3 directory
step3_dir = os.path.join(os.path.dirname(__file__))
os.chdir(step3_dir)

# Add the src directory to Python path
src_path = os.path.join(step3_dir, '../src')
sys.path.insert(0, src_path)

print(f"🔧 Running from: {os.getcwd()}")
print(f"🔧 Added to path: {src_path}")

# Now import and run
from unet_models import main

if __name__ == "__main__":
    print("🧪 Testing Step 3: UNet Models (from project root)")
    print("=" * 50)
    
    try:
        main()
        print("\n🎉 All UNet models loaded successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
