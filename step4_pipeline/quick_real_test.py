#!/usr/bin/env python3
"""
Quick Test with Real Images - Single Image Pair
Simple test to see virtual try-on results with real avatar and garment.
"""

import torch
import sys
import os
from PIL import Image

# Add path to Step 3
step3_path = os.path.join(os.path.dirname(__file__), '..', 'step3_unet_models')
sys.path.append(step3_path)

def main():
    """Quick test with one real image pair"""
    print("🎨 Quick Real Image Test - Virtual Try-On")
    print("=" * 50)
    
    try:
        # Import pipeline
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        print("🚀 Creating pipeline...")
        pipeline = SimpleTryOnPipeline()
        print("✅ Pipeline created successfully")
        
        # Test with one image pair
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
        print(f"\n📸 Loading images...")
        human_img = Image.open(human_path).convert('RGB')
        garment_img = Image.open(garment_path).convert('RGB')
        
        print(f"✅ Human: {human_img.size} ({human_img.mode})")
        print(f"✅ Garment: {garment_img.size} ({garment_img.mode})")
        
        # Generate try-on
        print(f"\n🎭 Generating virtual try-on...")
        print(f"   Prompt: A person wearing the casual top")
        print(f"   Steps: 10")
        
        result = pipeline.generate_tryon(
            human_image=human_img,
            garment_image=garment_img,
            prompt="A person wearing the casual top",
            num_inference_steps=10
        )
        
        print(f"✅ Try-on generated successfully!")
        print(f"   Result size: {result.size}")
        print(f"   Result mode: {result.mode}")
        
        # Save result
        output_path = "quick_tryon_result.png"
        result.save(output_path)
        print(f"💾 Result saved to: {output_path}")
        
        # Create simple comparison
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
        
        comparison_path = "quick_comparison.png"
        comparison.save(comparison_path)
        print(f"💾 Comparison saved to: {comparison_path}")
        
        # Cleanup
        pipeline.cleanup()
        print("\n🧹 Pipeline cleanup completed")
        
        print(f"\n🎉 Quick test completed successfully!")
        print(f"📁 Generated files:")
        print(f"   • {output_path} - Try-on result")
        print(f"   • {comparison_path} - Side-by-side comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
