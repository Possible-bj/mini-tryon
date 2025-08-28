# Step 2: Core Models

## 🎯 Goal
Learn how to load the essential AI models needed for virtual try-on.

## 📋 What We'll Do
1. Load VAE (Variational Autoencoder) - encodes/decodes images
2. Load Text Encoders (2x) - process text prompts
3. Load Scheduler - controls the diffusion process
4. Test basic model operations

## 🧠 Models We'll Load
- **VAE**: Converts images to/from latent space
- **Text Encoder 1**: CLIP text encoder
- **Text Encoder 2**: CLIP text encoder with projection
- **Scheduler**: DDPM scheduler for diffusion

## 🧪 Test
Run `python test_step2.py` to verify all models load correctly.

## ⚠️ Note
This step requires downloading models (~2GB). First run will be slower.
