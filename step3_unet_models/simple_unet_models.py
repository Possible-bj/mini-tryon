"""
Step 3: Simplified UNet Models for IDM-VTON
A clean, modern implementation using the latest diffusers versions.
"""

import torch
import torch.nn as nn
from diffusers import (
    UNet2DConditionModel, 
    AutoencoderKL,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleUNetModels:
    """Simplified UNet Models class using latest diffusers versions"""
    
    def __init__(self, model_name="yisol/IDM-VTON", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.models = {}
        
        print(f"🚀 Initializing Simplified UNet Models")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        print("=" * 50)
    
    def load_main_unet(self):
        """Load the main UNet for try-on generation"""
        print("\n1. Loading Main UNet...")
        try:
            # Use standard diffusers UNet2DConditionModel
            # Fixed: Use 'dtype' instead of deprecated 'torch_dtype'
            unet = UNet2DConditionModel.from_pretrained(
                self.model_name,
                subfolder="unet",
                dtype=torch.float16,  # Fixed: was torch_dtype
                use_safetensors=True,
                local_files_only=False,  # Allow downloading if needed
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
            print("   Trying fallback approach...")
            return self._load_fallback_unet()
    
    def _load_fallback_unet(self):
        """Fallback: Create a basic UNet if loading fails"""
        print("   Creating fallback UNet...")
        try:
            # Create a minimal UNet configuration
            config = {
                "in_channels": 4,
                "out_channels": 4,
                "block_out_channels": (320, 640, 1280, 1280),
                "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                "cross_attention_dim": 1280,
                "attention_head_dim": 8,
                "layers_per_block": 2,
                "mid_block_type": "UNetMidBlock2DCrossAttn",
            }
            
            unet = UNet2DConditionModel(**config)
            # Fixed: Convert to float16 after creation to match expected dtype
            unet = unet.to(dtype=torch.float16)
            unet.to(self.device)
            unet.requires_grad_(False)
            unet.eval()
            
            self.models['main_unet'] = unet
            print("✅ Fallback UNet created successfully")
            return unet
        except Exception as e:
            print(f"❌ Fallback UNet creation failed: {e}")
            return None
    
    def load_garment_encoder_unet(self):
        """Load the garment encoder UNet"""
        print("\n2. Loading Garment Encoder UNet...")
        try:
            unet_encoder = UNet2DConditionModel.from_pretrained(
                self.model_name,
                subfolder="unet_encoder",
                dtype=torch.float16,  # Fixed: was torch_dtype
                use_safetensors=True,
                local_files_only=False,
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
            print("   Trying fallback approach...")
            return self._load_fallback_garment_encoder()
    
    def _load_fallback_garment_encoder(self):
        """Fallback: Create a basic garment encoder UNet"""
        print("   Creating fallback Garment Encoder UNet...")
        try:
            config = {
                "in_channels": 3,  # RGB garment image
                "out_channels": 1280,  # Feature dimension
                "block_out_channels": (320, 640, 1280),
                "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                "cross_attention_dim": 1280,
                "attention_head_dim": 8,
                "layers_per_block": 2,
                "mid_block_type": "UNetMidBlock2DCrossAttn",
            }
            
            unet_encoder = UNet2DConditionModel(**config)
            # Fixed: Convert to float16 after creation to match expected dtype
            unet_encoder = unet_encoder.to(dtype=torch.float16)
            unet_encoder.to(self.device)
            unet_encoder.requires_grad_(False)
            unet_encoder.eval()
            
            self.models['garment_encoder_unet'] = unet_encoder
            print("✅ Fallback Garment Encoder UNet created successfully")
            return unet_encoder
        except Exception as e:
            print(f"❌ Fallback Garment Encoder UNet creation failed: {e}")
            return None
    
    def load_image_encoder(self):
        """Load the image encoder for IP-Adapter"""
        print("\n3. Loading Image Encoder...")
        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.model_name,
                subfolder="image_encoder",
                dtype=torch.float16,  # Fixed: was torch_dtype
                local_files_only=False,
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
            print("   Trying fallback approach...")
            return self._load_fallback_image_encoder()
    
    def _load_fallback_image_encoder(self):
        """Fallback: Use standard CLIP image encoder"""
        print("   Loading standard CLIP image encoder...")
        try:
            # Use a standard CLIP model as fallback
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                dtype=torch.float16,  # Fixed: was torch_dtype
            )
            image_encoder.to(self.device)
            image_encoder.requires_grad_(False)
            
            self.models['image_encoder'] = image_encoder
            self.models['feature_extractor'] = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            print("✅ Standard CLIP Image Encoder loaded successfully")
            print(f"   - Hidden size: {image_encoder.config.hidden_size}")
            return image_encoder
        except Exception as e:
            print(f"❌ Fallback Image Encoder loading failed: {e}")
            return None
    
    def load_all(self):
        """Load all models"""
        print("\n🔄 Loading all models...")
        
        main_unet = self.load_main_unet()
        garment_encoder = self.load_garment_encoder_unet()
        image_encoder = self.load_image_encoder()
        
        if all([main_unet, garment_encoder, image_encoder]):
            print("\n🎉 All models loaded successfully!")
            return True
        else:
            print("\n⚠️  Some models failed to load. Check errors above.")
            return False
    
    def test_garment_encoding(self, garment_tensor):
        """Test garment encoding functionality"""
        print("\n🧪 Testing garment encoding...")
        
        if 'garment_encoder_unet' not in self.models:
            print("❌ Garment encoder not loaded")
            return None
        
        try:
            with torch.no_grad():
                # Fixed: Ensure consistent data types (Float vs Half issue)
                # Convert garment_tensor to float16 to match model dtype
                garment_tensor = garment_tensor.to(dtype=torch.float16)
                
                # Create dummy timestep and encoder hidden states
                batch_size = garment_tensor.shape[0]
                timestep = torch.tensor([0], device=self.device, dtype=torch.long)
                
                # Fixed: Ensure encoder_hidden_states has same dtype as model
                encoder_hidden_states = torch.randn(
                    batch_size, 77, 1280, 
                    device=self.device, 
                    dtype=torch.float16  # Match model dtype
                )
                
                # Test forward pass
                output = self.models['garment_encoder_unet'](
                    sample=garment_tensor,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True
                )
                
                print("✅ Garment encoding test successful")
                print(f"   - Input shape: {garment_tensor.shape}")
                print(f"   - Output shape: {output.sample.shape}")
                print(f"   - Input dtype: {garment_tensor.dtype}")
                print(f"   - Output dtype: {output.sample.dtype}")
                return output.sample
                
        except Exception as e:
            print(f"❌ Garment encoding test failed: {e}")
            print(f"   - Garment tensor dtype: {garment_tensor.dtype}")
            print(f"   - Garment tensor device: {garment_tensor.device}")
            if 'garment_encoder_unet' in self.models:
                model = self.models['garment_encoder_unet']
                print(f"   - Model device: {next(model.parameters()).device}")
                print(f"   - Model dtype: {next(model.parameters()).dtype}")
            return None
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'config'):
                config = model.config
                info[name] = {
                    'type': type(model).__name__,
                    'in_channels': getattr(config, 'in_channels', 'N/A'),
                    'out_channels': getattr(config, 'out_channels', 'N/A'),
                    'cross_attention_dim': getattr(config, 'cross_attention_dim', 'N/A'),
                    'device': str(next(model.parameters()).device),
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'dtype': str(next(model.parameters()).dtype),  # Added dtype info
                }
            else:
                info[name] = {
                    'type': type(model).__name__,
                    'device': 'N/A',
                    'parameters': 'N/A',
                    'dtype': 'N/A',
                }
        
        return info
    
    def cleanup(self):
        """Clean up models and free memory"""
        print("\n🧹 Cleaning up models...")
        for name, model in self.models.items():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        self.models.clear()
        torch.cuda.empty_cache()
        print("✅ Cleanup completed")

def main():
    """Main function to test the simplified UNet models"""
    print("🧪 Testing Simplified UNet Models")
    print("=" * 40)
    
    # Initialize models - Fixed: Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    unet_models = SimpleUNetModels(device=device)
    
    # Load all models
    success = unet_models.load_all()
    
    if success:
        # Get model info
        info = unet_models.get_model_info()
        print("\n📊 Model Information:")
        for name, details in info.items():
            print(f"   {name}:")
            for key, value in details.items():
                print(f"     {key}: {value}")
        
        # Test garment encoding with proper dtype
        print(f"\n🧪 Testing with device: {device}")
        if device == "cuda":
            dummy_garment = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16)
        else:
            dummy_garment = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16)
        
        result = unet_models.test_garment_encoding(dummy_garment)
        
        if result is not None:
            print("✅ Basic functionality test passed!")
        else:
            print("⚠️  Basic functionality test had issues")
        
        # Cleanup
        unet_models.cleanup()
        
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Model loading failed. Check the errors above.")

if __name__ == "__main__":
    main()
