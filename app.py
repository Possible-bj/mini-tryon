import torch
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List
import os
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


class ChangeClothesAI:
    def __init__(self, base_path='yisol/IDM-VTON'):
        """Initialize the Change Clothes AI service"""
        self.base_path = base_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load all models
        self._load_models()
        self._setup_pipeline()
        
    def _load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.unet.requires_grad_(False)
        
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_path, 
            subfolder="scheduler"
        )

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        # Initialize preprocessing models
        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

        # Set requires_grad to False for all models
        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        print("Models loaded successfully!")

    def _setup_pipeline(self):
        """Setup the try-on pipeline"""
        print("Setting up pipeline...")
        
        self.pipe = TryonPipeline.from_pretrained(
            self.base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = self.UNet_Encoder
        
        # Move to device
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        
        print("Pipeline setup complete!")

    def try_on(self, human_img_path, garment_img_path, garment_description, 
                category="upper_body", denoise_steps=30, seed=None, 
                auto_mask=True, auto_crop=False, save_output=True, output_path="output"):
        """Main try-on function
        
        Args:
            human_img_path: Path to human image
            garment_img_path: Path to garment image  
            garment_description: Text description of the garment
            category: Clothing category (upper_body, lower_body, dresses)
            denoise_steps: Number of denoising steps
            seed: Random seed for generation
            auto_mask: Whether to use automatic masking
            auto_crop: Whether to auto-crop the image
            save_output: Whether to save the output
            output_path: Directory to save outputs
            
        Returns:
            tuple: (generated_image, mask_image)
        """
        print(f"Starting try-on process...")
        print(f"Human image: {human_img_path}")
        print(f"Garment image: {garment_img_path}")
        print(f"Category: {category}")
        
        # Load images
        human_img_orig = Image.open(human_img_path).convert("RGB")
        garm_img = Image.open(garment_img_path).convert("RGB")
        
        # Resize garment to standard size
        garm_img = garm_img.resize((768, 1024))
        orig_size = human_img_orig.size
        
        # Handle cropping if requested
        if auto_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))
        
        # Generate mask
        if auto_mask:
            print("Generating automatic mask...")
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            print("Using default mask...")
            # Create a simple default mask for upper body
            mask = Image.new('L', (768, 1024), 0)
            # Fill upper half with white (this is a simple approach)
            mask.paste(255, (0, 0, 768, 512))
        
        # Generate gray mask
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        
        # Generate pose information
        print("Generating pose information...")
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        args = apply_net.create_argument_parser().parse_args((
            'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
            './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', 
            '--opts', 'MODEL.DEVICE', self.device
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        
        # AI generation
        print("Generating AI image...")
        with torch.no_grad():
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                # Generate prompts
                prompt = "((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), the model is wearing " + garment_description
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch"
                
                # Encode prompts
                (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, 
                 negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt
                )
                
                # Generate garment prompts
                prompt = "((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), a photo of " + garment_description
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                
                (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt
                )
                
                # Prepare tensors
                pose_img_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
                garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(self.device, torch.float16)
                generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                
                # Generate image
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_tensor.to(self.device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=garm_tensor.to(self.device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]
        
        # Post-process results
        if auto_crop:
            result_img = images[0].resize(crop_size)
            result_mask = mask_gray.resize(crop_size)
        else:
            result_img = images[0].resize(orig_size)
            result_mask = mask_gray.resize(orig_size)
        
        # Save results if requested
        if save_output:
            os.makedirs(output_path, exist_ok=True)
            result_img.save(os.path.join(output_path, "generated_image.png"))
            result_mask.save(os.path.join(output_path, "mask_image.png"))
            print(f"Results saved to {output_path}/")
        
        print("Try-on process completed!")
        return result_img, result_mask


def test_try_on():
    """Test function to run the try-on service"""
    print("=== Change Clothes AI Test ===")
    
    # Initialize the service
    try:
        service = ChangeClothesAI()
        print("Service initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize service: {e}")
        return
    
    # Test with example images
    example_path = os.path.join(os.path.dirname(__file__), 'example')
    
    # Find example images
    human_examples = [f for f in os.listdir(os.path.join(example_path, "human")) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    garment_examples = [f for f in os.listdir(os.path.join(example_path, "cloth")) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not human_examples or not garment_examples:
        print("No example images found!")
        return
    
    # Use first available examples
    human_img_path = os.path.join(example_path, "human", human_examples[0])
    garment_img_path = os.path.join(example_path, "cloth", garment_examples[0])
    
    print(f"Testing with:")
    print(f"  Human: {human_examples[0]}")
    print(f"  Garment: {garment_examples[0]}")
    
    # Run try-on
    try:
        result_img, result_mask = service.try_on(
            human_img_path=human_img_path,
            garment_img_path=garment_img_path,
            garment_description="a blue t-shirt",
            category="upper_body",
            denoise_steps=20,  # Reduced for faster testing
            seed=42,
            auto_mask=True,
            auto_crop=False,
            save_output=True,
            output_path="test_output"
        )
        
        print("Test completed successfully!")
        print("Generated image and mask saved to test_output/ folder")
        
    except Exception as e:
        print(f"Try-on failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_try_on()
