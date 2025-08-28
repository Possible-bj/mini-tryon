"""
Step 3: UNet Models for IDM-VTON
Learn how to load the UNet models that do the actual image generation.
"""

import torch
import sys
import os
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# Add parent directory to path to access custom UNet models
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import custom UNet models
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

class UNetModels:
    """Class to manage UNet models for IDM-VTON"""
    
    def __init__(self, model_name="yisol/IDM-VTON", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.models = {}
        
        print(f"🚀 Initializing UNet Models")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        print("=" * 50)
    
    def load_main_unet(self):
        """Load the main UNet for try-on generation"""
        print("\n1. Loading Main UNet...")
        try:
            unet = UNet2DConditionModel.from_pretrained(
                self.model_name,
                subfolder="unet",
                torch_dtype=torch.float16,
            )
            unet.to(self.device)
            unet.requires_grad_(False)
            unet.eval()
            
            self.models['main_unet'] = unet
            print("✅ Main UNet loaded successfully")
            print(f"   - Generates the final try-on result")
            print(f"   - Uses human image + garment features + text prompts")
            print(f"   - Input channels: {unet.config.in_channels}")
            print(f"   - Cross attention dim: {unet.config.cross_attention_dim}")
            return unet
        except Exception as e:
            print(f"❌ Error loading Main UNet: {e}")
            return None
    
    def load_garment_encoder_unet(self):
        """Load the garment encoder UNet"""
        print("\n2. Loading Garment Encoder UNet...")
        try:
            unet_encoder = UNet2DConditionModel_ref.from_pretrained(
                self.model_name,
                subfolder="unet_encoder",
                torch_dtype=torch.float16,
            )
            unet_encoder.to(self.device)
            unet_encoder.requires_grad_(False)
            unet_encoder.eval()
            
            self.models['garment_encoder_unet'] = unet_encoder
            print("✅ Garment Encoder UNet loaded successfully")
            print(f"   - Processes garment images to extract features")
            print(f"   - Provides garment features to main UNet")
            print(f"   - Input channels: {unet_encoder.config.in_channels}")
            return unet_encoder
        except Exception as e:
            print(f"❌ Error loading Garment Encoder UNet: {e}")
            return None
    
    def load_image_encoder(self):
        """Load the image encoder for IP-Adapter"""
        print("\n3. Loading Image Encoder...")
        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.model_name,
                subfolder="image_encoder",
                torch_dtype=torch.float16,
            )
            image_encoder.to(self.device)
            image_encoder.requires_grad_(False)
            
            self.models['image_encoder'] = image_encoder
            self.models['feature_extractor'] = CLIPImageProcessor()
            
            print("✅ Image Encoder loaded successfully")
            print(f"   - Processes garment images for IP-Adapter")
            print(f"   - Provides additional garment conditioning")
            print(f"   - Hidden size: {image_encoder.config.hidden_size}")
            return image_encoder
        except Exception as e:
            print(f"❌ Error loading Image Encoder: {e}")
            return None
    
    def load_all(self):
        """Load all UNet models"""
        print("🔄 Loading all UNet models...")
        
        main_unet = self.load_main_unet()
        garment_encoder = self.load_garment_encoder_unet()
        image_encoder = self.load_image_encoder()
        
        if all([main_unet, garment_encoder, image_encoder]):
            print("\n🎉 All UNet models loaded successfully!")
            return True
        else:
            print("\n❌ Some UNet models failed to load")
            return False
    
    def test_garment_encoding(self, garment_tensor):
        """Test garment encoding functionality"""
        print(f"\n🧪 Testing garment encoding...")
        
        if 'garment_encoder_unet' not in self.models:
            print("❌ Garment encoder UNet not loaded")
            return None
        
        try:
            garment_encoder = self.models['garment_encoder_unet']
            
            # Create dummy timestep and text embeddings
            timestep = torch.tensor([0], device=self.device)
            text_embeds = torch.randn(1, 77, 768, device=self.device, dtype=torch.float16)
            
            with torch.no_grad():
                # Test garment encoding
                down_features, reference_features = garment_encoder(
                    garment_tensor, 
                    timestep, 
                    text_embeds,
                    return_dict=False
                )
            
            print(f"✅ Garment encoding successful")
            print(f"   - Input garment shape: {garment_tensor.shape}")
            print(f"   - Down features: {len(down_features)} layers")
            print(f"   - Reference features: {len(reference_features)} layers")
            
            return down_features, reference_features
            
        except Exception as e:
            print(f"❌ Error in garment encoding: {e}")
            return None
    
    def get_model_info(self):
        """Get information about loaded UNet models"""
        print("\n📊 UNet Model Information:")
        print("=" * 35)
        
        for name, model in self.models.items():
            if hasattr(model, '__class__'):
                print(f"   {name}: {model.__class__.__name__}")
                if hasattr(model, 'config'):
                    print(f"      - Input channels: {getattr(model.config, 'in_channels', 'N/A')}")
                    print(f"      - Cross attention dim: {getattr(model.config, 'cross_attention_dim', 'N/A')}")
            else:
                print(f"   {name}: {type(model).__name__}")

def main():
    """Main function to test UNet models"""
    print("🚀 Step 3: UNet Models for IDM-VTON")
    print("=" * 50)
    
    # Initialize UNet models
    unet_models = UNetModels()
    
    # Load all models
    success = unet_models.load_all()
    
    if success:
        # Create dummy garment tensor for testing
        print("\n🧪 Creating dummy garment tensor for testing...")
        dummy_garment = torch.randn(1, 3, 64, 48, device=unet_models.device, dtype=torch.float16)
        print(f"   Dummy garment shape: {dummy_garment.shape}")
        
        # Test garment encoding
        unet_models.test_garment_encoding(dummy_garment)
        
        # Show model info
        unet_models.get_model_info()
        
        print("\n✅ Step 3 completed successfully!")
        print("📝 Next: We'll combine everything into a pipeline in Step 4")
    else:
        print("\n❌ Step 3 failed - some UNet models didn't load")

if __name__ == "__main__":
    main()
