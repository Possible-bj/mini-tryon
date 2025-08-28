"""
Step 1: Basic Setup for IDM-VTON
Learn how to load and preprocess images for virtual try-on.
"""

import sys
import os
from PIL import Image
import numpy as np

# Add parent directory to path to access example images
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def load_image(image_path):
    """Load an image from file path"""
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Loaded image: {image_path}")
        print(f"   Size: {image.size}")
        return image
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

def resize_image(image, target_size=(768, 1024)):
    """Resize image to target size (width, height)"""
    resized = image.resize(target_size, Image.LANCZOS)
    print(f"✅ Resized image to: {target_size}")
    return resized

def create_simple_mask(image_size=(768, 1024)):
    """
    Create a simple rectangular mask for upper body region
    This is a basic mask - in later steps we'll use AI models for precise masking
    """
    width, height = image_size
    
    # Create a black mask
    mask = Image.new('L', (width, height), 0)
    
    # Define upper body region (rough approximation)
    upper_start = int(height * 0.2)  # Start at 20% from top
    upper_end = int(height * 0.7)    # End at 70% from top
    
    # Fill the upper body region with white (255)
    for y in range(upper_start, upper_end):
        for x in range(width):
            mask.putpixel((x, y), 255)
    
    print(f"✅ Created simple mask for upper body region")
    print(f"   Upper body: y={upper_start} to y={upper_end}")
    return mask

def save_image(image, output_path):
    """Save image to file"""
    try:
        image.save(output_path)
        print(f"✅ Saved image to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving image: {e}")

def main():
    """Main function to test basic setup"""
    print("🚀 Step 1: Basic Setup for IDM-VTON")
    print("=" * 50)
    
    # Try to load example images
    example_human = "../../gradio_demo/example/human/00121_00.jpg"
    example_garment = "../../gradio_demo/example/cloth/09305_00.jpg"
    
    # Load human image
    print("\n1. Loading human image...")
    human_img = load_image(example_human)
    if human_img is None:
        print("❌ Could not load human image. Please check the path.")
        return
    
    # Load garment image
    print("\n2. Loading garment image...")
    garment_img = load_image(example_garment)
    if garment_img is None:
        print("❌ Could not load garment image. Please check the path.")
        return
    
    # Resize images
    print("\n3. Resizing images...")
    human_resized = resize_image(human_img)
    garment_resized = resize_image(garment_img)
    
    # Create mask
    print("\n4. Creating simple mask...")
    mask = create_simple_mask()
    
    # Save results
    print("\n5. Saving results...")
    save_image(human_resized, "human_resized.jpg")
    save_image(garment_resized, "garment_resized.jpg")
    save_image(mask, "simple_mask.jpg")
    
    print("\n✅ Step 1 completed successfully!")
    print("📁 Check the generated files:")
    print("   - human_resized.jpg")
    print("   - garment_resized.jpg") 
    print("   - simple_mask.jpg")

if __name__ == "__main__":
    main()
