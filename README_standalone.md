# Change Clothes AI - Standalone Service

This is a standalone version of the Change Clothes AI service, converted from the Hugging Face Spaces app to work as a local service or API.

## What it does

The service uses AI to virtually try on clothing items on human images. It combines:
- **OpenPose** for body keypoint detection
- **DensePose** for body segmentation  
- **Human Parsing** for body part segmentation
- **Stable Diffusion XL** for realistic image generation

## Files

- `app.py` - Main service class (`ChangeClothesAI`)
- `test_service.py` - Simple test script
- `api_service.py` - Flask API service for frontend integration
- `requirements_standalone.txt` - Dependencies without Gradio/spaces

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_standalone.txt
   ```

2. **Additional dependencies for API:**
   ```bash
   pip install flask flask-cors
   ```

## Quick Test

Run the test script to verify everything works:

```bash
python test_service.py
```

This will:
- Use example images from the `example/` folder
- Run the try-on process
- Save results to `test_output/` folder
- Display progress and results

## Using the Service

### As a Python Class

```python
from app import ChangeClothesAI

# Initialize service
service = ChangeClothesAI()

# Run try-on
result_img, result_mask = service.try_on(
    human_img_path="path/to/human.jpg",
    garment_img_path="path/to/garment.jpg", 
    garment_description="a blue t-shirt",
    category="upper_body",
    denoise_steps=30,
    seed=42,
    auto_mask=True,
    auto_crop=False,
    save_output=True,
    output_path="output"
)
```

### As an API Service

1. **Start the API:**
   ```bash
   python api_service.py
   ```

2. **Test with curl:**
   ```bash
   curl -X POST \
     -F 'human_image=@human.jpg' \
     -F 'garment_image=@garment.jpg' \
     -F 'garment_description=a blue t-shirt' \
     -F 'category=upper_body' \
     http://localhost:5000/try-on
   ```

3. **API Endpoints:**
   - `GET /health` - Health check
   - `POST /try-on` - Try-on with base64 response
   - `POST /try-on-file` - Try-on with file response

## Parameters

- **human_img_path**: Path to human image
- **garment_img_path**: Path to garment image
- **garment_description**: Text description of the garment
- **category**: `upper_body`, `lower_body`, or `dresses`
- **denoise_steps**: Number of denoising steps (20-40, default: 30)
- **seed**: Random seed for reproducible results
- **auto_mask**: Whether to use automatic masking
- **auto_crop**: Whether to auto-crop to 3:4 ratio
- **save_output**: Whether to save results to disk
- **output_path**: Directory to save outputs

## Example Images

The service comes with example images in the `example/` folder:
- `example/human/` - Sample human images
- `example/cloth/` - Sample garment images

## Output

The service generates:
1. **Generated Image**: The final result showing the person wearing the garment
2. **Mask Image**: The mask used during generation

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `denoise_steps` or use CPU mode
2. **Model loading errors**: Check if all model files are present
3. **Image format errors**: Ensure images are JPG/PNG format

### Performance Tips

- Use GPU if available (automatically detected)
- Reduce `denoise_steps` for faster generation
- Use `auto_crop=True` for better results with certain images

## Next Steps

Once you've tested the service:

1. **Customize parameters** for your use case
2. **Integrate with your frontend** using the API service
3. **Deploy to production** (consider using Gunicorn for the Flask app)
4. **Add authentication** and rate limiting as needed

## API Integration Example

```javascript
// Frontend JavaScript example
const formData = new FormData();
formData.append('human_image', humanFile);
formData.append('garment_image', garmentFile);
formData.append('garment_description', 'a stylish t-shirt');

fetch('http://localhost:5000/try-on', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    // Display generated image
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + data.generated_image;
    document.body.appendChild(img);
  }
});
```
