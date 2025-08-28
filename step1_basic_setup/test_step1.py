"""
Test script for Step 1: Basic Setup
Run this to verify everything works correctly.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from basic_setup import main

if __name__ == "__main__":
    print("🧪 Testing Step 1: Basic Setup")
    print("=" * 40)
    
    try:
        main()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
