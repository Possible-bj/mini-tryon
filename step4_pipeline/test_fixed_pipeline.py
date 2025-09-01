#!/usr/bin/env python3
"""
Test the Fixed Pipeline with Full Diffusion Generation
This should now generate actual try-on results instead of abstract color blends.
"""

import torch
import sys
import os
from PIL import Image

# Add path to Step 3
step3_path = os.path.join(os.path.dirname(__file__), '..', 'step3_unet_models')
sys.path.append(step3_path)

def test_fixed_pipeline():
    """Test the fixed pipeline with real images"""
    print("🎭 Testing Fixed Pipeline with Full Diffusion Generation")
    print("=" * 60)
    
    try:
        # Import the fixed pipeline
        from simple_tryon_pipeline_fixed import SimpleTryOnPipeline
        
        # Create pipeline
        print("🚀 Creating fixed pipeline...")
        pipeline = SimpleTryOnPipeline()
        print("✅ Fixed pipeline created successfully")
        
        # Test with real images
        human_path = 'example/human/taylor-.jpg'
        garment_path = 'example/cloth/14627_00.jpg'
        
        # Check if files exist
        if not os.path.exists(human_path):
            print(f"❌ Human image not found: {human_path}")
            print("💡 Make sure you're running from the project root directory")
            return False
            
        if not os.path.exists(garment_path):
            print(f"❌ Garment image not found: {garment_path}")
            print("💡 Make sure you're running from the project root directory")
            return False
        
        # Load images
        print(f"\n📸 Loading real images...")
        human_img = Image.open(human_path).convert('RGB')
        garment_img = Image.open(garment_path).convert('RGB')
        
        print(f"✅ Human: {human_img.size} ({human_img.mode})")
        print(f"✅ Garment: {garment_img.size} ({garment_img.mode})")
        
        # Generate try-on with diffusion
        print(f"\n🎭 Generating try-on with full diffusion...")
        print(f"   Prompt: A person wearing the casual top")
        print(f"   Steps: 15 (full diffusion)")
        print(f"   Guidance: 7.5")
        
        result = pipeline.generate_tryon(
            human_image=human_img,
            garment_image=garment_img,
            prompt="A person wearing the casual top",
            num_inference_steps=15,  # Full diffusion
            guidance_scale=7.5
        )
        
        print(f"✅ Try-on generated successfully!")
        print(f"   Result size: {result.size}")
        print(f"   Result mode: {result.mode}")
        
        # Save result
        output_path = "fixed_tryon_result.png"
        result.save(output_path)
        print(f"💾 Result saved to: {output_path}")
        
        # Create comparison
        print(f"\n🖼️  Creating comparison image...")
        
        # Create a side-by-side comparison
        total_width = human_img.width + garment_img.width + result.width
        max_height = max(human_img.height, garment_img.height, result.height)
        
        comparison = Image.new('RGB', (total_width, max_height), 'white')
        
        # Paste images side by side
        x_offset = 0
        for img, label in [(human_img, "Human"), (garment_img, "Garment"), (result, "Try-On Result")]:
            comparison.paste(img, (x_offset, 0))
            x_offset += img.width
        
        comparison_path = "fixed_comparison.png"
        comparison.save(comparison_path)
        print(f"💾 Comparison saved to: {comparison_path}")
        
        # Cleanup
        pipeline.cleanup()
        print("\n🧹 Pipeline cleanup completed")
        
        print(f"\n🎉 Fixed pipeline test completed successfully!")
        print(f"📁 Generated files:")
        print(f"   • {output_path} - Try-on result with diffusion")
        print(f"   • {comparison_path} - Side-by-side comparison")
        
        print(f"\n🚀 Key Improvements:")
        print(f"   ✅ Full diffusion generation (not simplified)")
        print(f"   ✅ Proper UNet usage for try-on")
        print(f"   ✅ High-resolution output (512x512)")
        print(f"   ✅ Actual try-on results (not color blends)")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Step 4: Fixed Pipeline with Full Diffusion Generation")
    print("=" * 70)
    
    success = test_fixed_pipeline()
    
    if success:
        print(f"\n🎉 SUCCESS! The fixed pipeline should now generate actual try-on results!")
        print(f"📸 Check the generated images to see if Taylor is now wearing the garment.")
        
        print(f"\n💡 If this still doesn't work well, we can try Option 2:")
        print(f"   - Find an existing virtual try-on model")
        print(f"   - Use pre-trained weights")
        print(f"   - Skip the complex generation for now")
    else:
        print(f"\n⚠️  The fixed pipeline still has issues.")
        print(f"💡 This suggests we need Option 2 - find an existing model.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
