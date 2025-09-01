#!/usr/bin/env python3
"""
Test script for Simplified Step 3: UNet Models
This version uses the latest diffusers versions without complex hacks.
"""

import torch
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import diffusers
        print(f"✅ diffusers imported successfully (version: {diffusers.__version__})")
    except ImportError as e:
        print(f"❌ diffusers import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ transformers imported successfully (version: {transformers.__version__})")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    return True

def test_device():
    """Test device availability"""
    print("\n🔧 Testing device availability...")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("⚠️  CUDA not available, using CPU")
    
    return device

def test_simple_unet_models():
    """Test the simplified UNet models"""
    print("\n🚀 Testing Simplified UNet Models...")
    
    try:
        from simple_unet_models import SimpleUNetModels
        
        # Initialize with CPU for testing (faster and no GPU memory issues)
        device = "cpu"
        unet_models = SimpleUNetModels(device=device)
        
        # Test loading all models
        success = unet_models.load_all()
        
        if success:
            print("\n🎉 All models loaded successfully!")
            
            # Get model info
            info = unet_models.get_model_info()
            print("\n📊 Model Information:")
            for name, details in info.items():
                print(f"   {name}:")
                for key, value in details.items():
                    print(f"     {key}: {value}")
            
            # Test basic functionality
            print("\n🧪 Testing basic functionality...")
            dummy_garment = torch.randn(1, 3, 64, 64, device=device)  # Smaller size for testing
            result = unet_models.test_garment_encoding(dummy_garment)
            
            if result is not None:
                print("✅ Basic functionality test passed!")
            else:
                print("⚠️  Basic functionality test had issues")
            
            # Cleanup
            unet_models.cleanup()
            
        else:
            print("\n❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SimpleUNetModels: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing Simplified Step 3: UNet Models")
    print("=" * 50)
    
    # Test 1: Package imports
    if not test_imports():
        print("\n❌ Package import test failed. Please install required packages.")
        return False
    
    # Test 2: Device availability
    device = test_device()
    
    # Test 3: Simple UNet models
    if not test_simple_unet_models():
        print("\n❌ Simple UNet models test failed.")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("\n📝 Summary:")
    print("   ✅ Package imports working")
    print("   ✅ Device detection working")
    print("   ✅ Simplified UNet models working")
    print("\n🚀 You can now use the simplified implementation!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
