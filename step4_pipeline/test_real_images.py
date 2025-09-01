#!/usr/bin/env python3
"""
Test Step 4 Pipeline with Real Images
Uses actual avatar and garment images to demonstrate virtual try-on results.
"""

import torch
import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add path to Step 3
step3_path = os.path.join(os.path.dirname(__file__), '..', 'step3_unet_models')
sys.path.append(step3_path)

def load_and_preview_images(human_path, garment_path):
    """Load and preview the input images"""
    print("📸 Loading real images...")
    
    try:
        # Load images
        human_img = Image.open(human_path).convert('RGB')
        garment_img = Image.open(garment_path).convert('RGB')
        
        print(f"✅ Human image loaded: {human_img.size} ({human_img.mode})")
        print(f"✅ Garment image loaded: {garment_img.size} ({garment_img.mode})")
        
        # Show image info
        print(f"\n📊 Image Details:")
        print(f"   Human: {human_img.size[0]}x{human_img.size[1]} pixels")
        print(f"   Garment: {garment_img.size[0]}x{garment_img.size[1]} pixels")
        
        return human_img, garment_img
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return None, None

def test_pipeline_with_real_images():
    """Test the pipeline with real avatar and garment images"""
    print("🎨 Testing Pipeline with Real Images")
    print("=" * 50)
    
    try:
        # Import pipeline
        from simple_tryon_pipeline import SimpleTryOnPipeline
        
        # Create pipeline
        print("🚀 Creating pipeline...")
        pipeline = SimpleTryOnPipeline()
        print("✅ Pipeline created successfully")
        
        # Test with different image combinations
        test_cases = [
            {
                'human': 'example/human/taylor-.jpg',
                'garment': 'example/cloth/14627_00.jpg',
                'description': 'Taylor wearing casual top'
            },
            {
                'human': 'example/human/will1 (1).jpg', 
                'garment': 'example/cloth/09290_00.jpg',
                'description': 'Will wearing formal shirt'
            },
            {
                'human': 'example/human/Jensen.jpeg',
                'garment': 'example/cloth/10165_00.jpg', 
                'description': 'Jensen wearing casual top'
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Test Case {i+1}: {test_case['description']}")
            print("-" * 40)
            
            # Check if files exist
            if not os.path.exists(test_case['human']):
                print(f"⚠️  Human image not found: {test_case['human']}")
                continue
            if not os.path.exists(test_case['garment']):
                print(f"⚠️  Garment image not found: {test_case['garment']}")
                continue
            
            # Load images
            human_img, garment_img = load_and_preview_images(
                test_case['human'], 
                test_case['garment']
            )
            
            if human_img is None or garment_img is None:
                continue
            
            # Generate try-on
            print(f"\n🎭 Generating try-on for: {test_case['description']}")
            try:
                result = pipeline.generate_tryon(
                    human_image=human_img,
                    garment_image=garment_img,
                    prompt=f"A person wearing the {test_case['description']}",
                    num_inference_steps=10
                )
                
                print(f"✅ Try-on generated successfully!")
                print(f"   Result size: {result.size}")
                print(f"   Result mode: {result.mode}")
                
                # Save result
                output_path = f"tryon_result_{i+1}.png"
                result.save(output_path)
                print(f"💾 Result saved to: {output_path}")
                
                results.append({
                    'human': human_img,
                    'garment': garment_img,
                    'result': result,
                    'description': test_case['description'],
                    'output_path': output_path
                })
                
            except Exception as e:
                print(f"❌ Try-on generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Display results summary
        print(f"\n📊 Results Summary:")
        print(f"   Successful try-ons: {len(results)}/{len(test_cases)}")
        
        if results:
            print(f"\n🎉 All successful results saved!")
            for result in results:
                print(f"   ✅ {result['description']}: {result['output_path']}")
        
        # Cleanup
        pipeline.cleanup()
        print("\n🧹 Pipeline cleanup completed")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_comparison_grid(results):
    """Create a comparison grid showing before/after images"""
    if not results:
        print("⚠️  No results to display")
        return
    
    print("\n🖼️  Creating comparison grid...")
    
    try:
        # Create subplot grid
        n_results = len(results)
        fig, axes = plt.subplots(n_results, 3, figsize=(15, 5*n_results))
        
        if n_results == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            # Human image
            axes[i, 0].imshow(result['human'])
            axes[i, 0].set_title(f'Human {i+1}')
            axes[i, 0].axis('off')
            
            # Garment image
            axes[i, 1].imshow(result['garment'])
            axes[i, 1].set_title(f'Garment {i+1}')
            axes[i, 1].axis('off')
            
            # Try-on result
            axes[i, 2].imshow(result['result'])
            axes[i, 2].set_title(f'Try-On Result {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save comparison grid
        comparison_path = "tryon_comparison_grid.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"💾 Comparison grid saved to: {comparison_path}")
        
        # Show plot (if in interactive environment)
        try:
            plt.show()
        except:
            print("📱 Plot saved but not displayed (non-interactive environment)")
        
    except Exception as e:
        print(f"❌ Error creating comparison grid: {e}")

def main():
    """Main function to test real images"""
    print("🎨 Step 4: Real Image Virtual Try-On Test")
    print("=" * 60)
    
    # Test pipeline with real images
    results = test_pipeline_with_real_images()
    
    if results:
        # Create comparison grid
        create_comparison_grid(results)
        
        print(f"\n🎉 Real image testing completed successfully!")
        print(f"📁 Check the current directory for generated images:")
        for result in results:
            print(f"   • {result['output_path']}")
        print(f"   • tryon_comparison_grid.png")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review the generated try-on images")
        print(f"   2. Test with your own images")
        print(f"   3. Move to Step 5 (preprocessing integration)")
    else:
        print(f"\n⚠️  No successful try-ons generated")
        print(f"💡 Check the error messages above for troubleshooting")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
