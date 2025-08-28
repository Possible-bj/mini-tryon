# Step 3: UNet Models

## 🎯 Goal
Learn how to load the UNet models that do the actual image generation for virtual try-on.

## 📋 What We'll Do
1. Load Main UNet - The primary image generator
2. Load Garment Encoder UNet - Processes garment features
3. Test basic UNet operations
4. Understand how the two UNets work together

## 🧠 Models We'll Load
- **Main UNet**: Generates the final try-on result
- **Garment Encoder UNet**: Extracts features from garment images
- **Image Encoder**: Processes garment images for IP-Adapter

## 🔗 How They Work Together
1. **Garment Encoder UNet** processes the garment image to extract features
2. **Main UNet** uses those features + human image + text prompts to generate the result
3. **Image Encoder** provides additional garment conditioning

## 🧪 Test
Run `python test_step3.py` to verify all UNet models load correctly.

## ⚠️ Note
This step requires the custom UNet models from the IDM-VTON repository.
