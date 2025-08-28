"""
Step 2: Core Models for IDM-VTON
Learn how to load the essential AI models.
"""

import torch
import sys
import os
from transformers import (
    AutoTokenizer, 
    CLIPTextModel, 
    CLIPTextModelWithProjection
)
from diffusers import AutoencoderKL, DDPMScheduler

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

class CoreModels:
    """Class to manage core models for IDM-VTON"""
    
    def __init__(self, model_name="yisol/IDM-VTON", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.models = {}
        
        print(f"🚀 Initializing Core Models")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        print("=" * 50)
    
    def load_vae(self):
        """Load VAE (Variational Autoencoder)"""
        print("\n1. Loading VAE...")
        try:
            vae = AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae",
                torch_dtype=torch.float16,
            )
            vae.to(self.device)
            vae.requires_grad_(False)
            
            self.models['vae'] = vae
            print("✅ VAE loaded successfully")
            print(f"   - Encodes images to latent space")
            print(f"   - Decodes latents back to images")
            return vae
        except Exception as e:
            print(f"❌ Error loading VAE: {e}")
            return None
    
    def load_text_encoders(self):
        """Load text encoders"""
        print("\n2. Loading Text Encoders...")
        
        # Load tokenizers
        print("   Loading tokenizers...")
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.model_name, 
            subfolder="tokenizer"
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.model_name, 
            subfolder="tokenizer_2"
        )
        
        # Load text encoders
        print("   Loading text encoders...")
        text_encoder_one = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.model_name,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        
        # Move to device and freeze
        text_encoder_one.to(self.device)
        text_encoder_two.to(self.device)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        
        self.models.update({
            'tokenizer_one': tokenizer_one,
            'tokenizer_two': tokenizer_two,
            'text_encoder_one': text_encoder_one,
            'text_encoder_two': text_encoder_two
        })
        
        print("✅ Text encoders loaded successfully")
        print(f"   - Tokenizer 1: {tokenizer_one.__class__.__name__}")
        print(f"   - Tokenizer 2: {tokenizer_two.__class__.__name__}")
        print(f"   - Text Encoder 1: {text_encoder_one.__class__.__name__}")
        print(f"   - Text Encoder 2: {text_encoder_two.__class__.__name__}")
        
        return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
    
    def load_scheduler(self):
        """Load scheduler"""
        print("\n3. Loading Scheduler...")
        try:
            scheduler = DDPMScheduler.from_pretrained(
                self.model_name, 
                subfolder="scheduler"
            )
            
            self.models['scheduler'] = scheduler
            print("✅ Scheduler loaded successfully")
            print(f"   - Type: {scheduler.__class__.__name__}")
            print(f"   - Controls diffusion process")
            return scheduler
        except Exception as e:
            print(f"❌ Error loading scheduler: {e}")
            return None
    
    def load_all(self):
        """Load all core models"""
        print("🔄 Loading all core models...")
        
        vae = self.load_vae()
        text_encoders = self.load_text_encoders()
        scheduler = self.load_scheduler()
        
        if all([vae, text_encoders, scheduler]):
            print("\n🎉 All core models loaded successfully!")
            return True
        else:
            print("\n❌ Some models failed to load")
            return False
    
    def test_text_encoding(self, prompt="model wearing a blue shirt"):
        """Test text encoding functionality"""
        print(f"\n🧪 Testing text encoding with prompt: '{prompt}'")
        
        if not all(key in self.models for key in ['tokenizer_one', 'text_encoder_one']):
            print("❌ Text encoders not loaded")
            return None
        
        try:
            # Tokenize
            tokenizer = self.models['tokenizer_one']
            text_encoder = self.models['text_encoder_one']
            
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Encode
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(self.device))
            
            print(f"✅ Text encoding successful")
            print(f"   - Input tokens: {text_inputs.input_ids.shape}")
            print(f"   - Output embeddings: {text_embeddings.last_hidden_state.shape}")
            
            return text_embeddings
            
        except Exception as e:
            print(f"❌ Error in text encoding: {e}")
            return None
    
    def get_model_info(self):
        """Get information about loaded models"""
        print("\n📊 Model Information:")
        print("=" * 30)
        
        for name, model in self.models.items():
            if hasattr(model, '__class__'):
                print(f"   {name}: {model.__class__.__name__}")
            else:
                print(f"   {name}: {type(model).__name__}")

def main():
    """Main function to test core models"""
    print("🚀 Step 2: Core Models for IDM-VTON")
    print("=" * 50)
    
    # Initialize core models
    core_models = CoreModels()
    
    # Load all models
    success = core_models.load_all()
    
    if success:
        # Test text encoding
        core_models.test_text_encoding()
        
        # Show model info
        core_models.get_model_info()
        
        print("\n✅ Step 2 completed successfully!")
        print("📝 Next: We'll load the UNet models in Step 3")
    else:
        print("\n❌ Step 2 failed - some models didn't load")

if __name__ == "__main__":
    main()
