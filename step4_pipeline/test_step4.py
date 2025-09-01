#!/usr/bin/env python3
"""
Test script for Step 4: Pipeline Integration
Tests the complete try-on pipeline with all models from Step 3.
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✅ PIL imported successfully (version: {PIL.__version__})")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy imported successfully (version: {numpy.__version__})")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_device():
    """Test device availability"""
    print("\n🔧 Testing device availability...")
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        print("⚠️  CUDA not available, using CPU")
    
    print(f"📱 Using device: {device}")
    return device

def test_step3_models():
    """Test if Step 3 models can be imported"""
    print("\n📦 Testing Step 3 model imports...")
    
    try:
        # Add path to Step 3
        step3_path = os.path.join(os.path.dirname(__file__), '..', 'step3_unet_models')
        sys.path.append(step3_path)
        
        from simple_unet_models import SimpleUNetModels
        print("✅ Step 3 models imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Step 3 models import failed: {e}")
        print("💡 Make sure Step 3 is completed successfully first")
        return False

def test_pipeline_creation():
    """Test pipeline creation"""
    print("\n🚀 Testing pipeline creation...")
    
    try:
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        pipeline = SimpleTryOnPipeline()
        print("✅ Pipeline created successfully")
        
        # Test basic attributes
        assert hasattr(pipeline, 'unet'), "Pipeline missing main UNet"
        assert hasattr(pipeline, 'unet_encoder'), "Pipeline missing garment encoder UNet"
        assert hasattr(pipeline, 'image_encoder'), "Pipeline missing image encoder"
        assert hasattr(pipeline, 'feature_extractor'), "Pipeline missing feature extractor"
        
        print("✅ All required models are accessible")
        
        # Cleanup
        pipeline.cleanup()
        print("✅ Pipeline cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing functionality"""
    print("\n🖼️  Testing image processing...")
    
    try:
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        pipeline = SimpleTryOnPipeline()
        
        # Create test images
        test_human = Image.new('RGB', (128, 128), color='red')
        test_garment = Image.new('RGB', (128, 128), color='blue')
        
        # Test preprocessing
        human_tensor = pipeline.preprocess_image(test_human, (64, 64))
        garment_tensor = pipeline.preprocess_image(test_garment, (64, 64))
        
        print(f"✅ Human image preprocessed: {human_tensor.shape}")
        print(f"✅ Garment image preprocessed: {garment_tensor.shape}")
        
        # Test encoding
        garment_features = pipeline.encode_garment(garment_tensor)
        human_features = pipeline.encode_human_image(human_tensor)
        
        print(f"✅ Garment encoded: {garment_features.shape}")
        print(f"✅ Human encoded: {human_features.shape}")
        
        # Cleanup
        pipeline.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline workflow"""
    print("\n🎨 Testing complete pipeline workflow...")
    
    try:
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        pipeline = SimpleTryOnPipeline()
        
        # Create test images
        test_human = Image.new('RGB', (64, 64), color='red')
        test_garment = Image.new('RGB', (64, 64), color='blue')
        
        # Test full generation
        result = pipeline.generate_tryon(
            human_image=test_human,
            garment_image=test_garment,
            prompt="A person wearing the blue garment",
            num_inference_steps=5
        )
        
        print(f"✅ Generation completed! Result: {result.size}")
        print(f"✅ Result mode: {result.mode}")
        
        # Cleanup
        pipeline.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Step 4: Pipeline Integration Tests")
    print("=" * 50)
    
    # Track test results
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Device
    device = test_device()
    tests_passed += 1  # Device test always passes (just shows info)
    
    # Test 3: Step 3 Models
    if test_step3_models():
        tests_passed += 1
    
    # Test 4: Pipeline Creation
    if test_pipeline_creation():
        tests_passed += 1
    
    # Test 5: Image Processing
    if test_image_processing():
        tests_passed += 1
    
    # Test 6: Full Pipeline
    if test_full_pipeline():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Step 4 is ready!")
        print("\n🚀 Next steps:")
        print("   1. Test with real images")
        print("   2. Integrate with preprocessing (Step 5)")
        print("   3. Build API interface (Step 6)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common issues:")
        print("   - Make sure Step 3 is completed successfully")
        print("   - Check that all dependencies are installed")
        print("   - Verify CUDA setup if using GPU")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
