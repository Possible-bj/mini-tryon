#!/usr/bin/env python3
"""
Simple test script for ChangeClothesAI service
"""

import os
from app import ChangeClothesAI

def test_with_custom_images():
    """Test the service with custom image paths"""
    
    # You can modify these paths to test with your own images
    human_img_path = "example/human/00034_00.jpg"  # Change this to your human image path
    garment_img_path = "example/cloth/14627_00.jpg"  # Change this to your garment image path
    
    # Check if images exist
    if not os.path.exists(human_img_path):
        print(f"Human image not found: {human_img_path}")
        print("Please provide a valid path to a human image")
        return
    
    if not os.path.exists(garment_img_path):
        print(f"Garment image not found: {garment_img_path}")
        print("Please provide a valid path to a garment image")
        return
    
    print("=== Testing ChangeClothesAI Service ===")
    print(f"Human image: {human_img_path}")
    print(f"Garment image: {garment_img_path}")
    
    try:
        # Initialize the service
        print("Initializing service...")
        service = ChangeClothesAI()
        print("Service initialized successfully!")
        
        # Run try-on
        print("Running try-on...")
        result_img, result_mask = service.try_on(
            human_img_path=human_img_path,
            garment_img_path=garment_img_path,
            garment_description="a stylish t-shirt",
            category="upper_body",
            denoise_steps=20,  # Reduced for faster testing
            seed=42,
            auto_mask=True,
            auto_crop=False,
            save_output=True,
            output_path="test_output"
        )
        
        print("✅ Try-on completed successfully!")
        print("📁 Results saved to 'test_output/' folder:")
        print("   - generated_image.png (the final result)")
        print("   - mask_image.png (the mask used)")
        
        # Display image info
        print(f"📊 Generated image size: {result_img.size}")
        print(f"📊 Mask image size: {result_mask.size}")
        
    except Exception as e:
        print(f"❌ Error during try-on: {e}")
        import traceback
        traceback.print_exc()

def test_with_example_images():
    """Test with the first available example images"""
    
    example_path = "example"
    if not os.path.exists(example_path):
        print("Example folder not found!")
        return
    
    # Find first available images
    human_dir = os.path.join(example_path, "human")
    garment_dir = os.path.join(example_path, "cloth")
    
    if not os.path.exists(human_dir) or not os.path.exists(garment_dir):
        print("Example subdirectories not found!")
        return
    
    human_files = [f for f in os.listdir(human_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    garment_files = [f for f in os.listdir(garment_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not human_files or not garment_files:
        print("No example images found!")
        return
    
    human_img_path = os.path.join(human_dir, human_files[0])
    garment_img_path = os.path.join(garment_dir, garment_files[0])
    
    print("=== Testing with Example Images ===")
    print(f"Human: {human_files[0]}")
    print(f"Garment: {garment_files[0]}")
    
    try:
        service = ChangeClothesAI()
        result_img, result_mask = service.try_on(
            human_img_path=human_img_path,
            garment_img_path=garment_img_path,
            garment_description="a casual t-shirt",
            category="upper_body",
            denoise_steps=20,
            seed=42,
            auto_mask=True,
            auto_crop=False,
            save_output=True,
            output_path="test_output"
        )
        
        print("✅ Test completed successfully!")
        print("📁 Results saved to 'test_output/' folder")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("ChangeClothesAI Service Test")
    print("=" * 40)
    
    # Try to test with custom images first, fall back to examples
    try:
        test_with_custom_images()
    except Exception as e:
        print(f"Custom image test failed: {e}")
        print("\nFalling back to example images...")
        test_with_example_images()
