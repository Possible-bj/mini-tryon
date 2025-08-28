"""
Test script for Step 2: Core Models
Run this to verify all core models load correctly.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core_models import main

if __name__ == "__main__":
    print("🧪 Testing Step 2: Core Models")
    print("=" * 40)
    
    try:
        main()
        print("\n🎉 All core models loaded successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
