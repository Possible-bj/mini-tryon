#!/usr/bin/env python3
"""
Quick test to verify Step 4 pipeline fixes
"""

import sys
import os

# Add path to Step 3
step3_path = os.path.join(os.path.dirname(__file__), '..', 'step3_unet_models')
sys.path.append(step3_path)

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        from simple_unet_models import SimpleUNetModels
        print("✅ Step 3 models imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Step 3 models import failed: {e}")
        return False

def test_pipeline_import():
    """Test pipeline import"""
    print("\n🚀 Testing pipeline import...")
    
    try:
        from simple_tryon_pipeline import SimpleTryOnPipeline
        print("✅ Pipeline imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation"""
    print("\n🔧 Testing pipeline creation...")
    
    try:
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        pipeline = SimpleTryOnPipeline()
        print("✅ Pipeline created successfully")
        
        # Check if models are accessible
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

def main():
    """Run quick tests"""
    print("🧪 Quick Test: Step 4 Pipeline Fixes")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic imports
    if test_basic_imports():
        tests_passed += 1
    
    # Test 2: Pipeline import
    if test_pipeline_import():
        tests_passed += 1
    
    # Test 3: Pipeline creation
    if test_pipeline_creation():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Pipeline fixes are working!")
        print("\n🚀 You can now run the full test suite:")
        print("   python test_step4.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
