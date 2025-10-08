#!/usr/bin/env python3
"""
Flask API service for ChangeClothesAI
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
from PIL import Image
import io
import base64
import requests
from urllib.parse import urlparse
import threading
from app import ChangeClothesAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global service instance and model status tracking
service = None
model_status = "not_loaded"  # "not_loaded", "loading", "loaded"
loading_lock = threading.Lock()

def get_service():
    """Get or create the ChangeClothesAI service instance"""
    global service, model_status
    
    # If already loaded, return immediately
    if service is not None and model_status == "loaded":
        return service
    
    # Use lock to prevent multiple simultaneous loading attempts
    with loading_lock:
        # Double-check after acquiring lock
        if service is not None and model_status == "loaded":
            return service
        
        # If already loading, return None (caller should check status)
        if model_status == "loading":
            return None
        
        # Start loading
        model_status = "loading"
        print("Initializing ChangeClothesAI service...")
        
        try:
            service = ChangeClothesAI()
            model_status = "loaded"
            print("Service initialized!")
            return service
        except Exception as e:
            # Reset status on error
            model_status = "not_loaded"
            service = None
            print(f"Failed to initialize service: {e}")
            raise e

def download_image_from_url(url, timeout=30):
    """Download image from URL and return PIL Image object"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Download image
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to an image (content-type: {content_type})")
        
        # Load image
        image = Image.open(io.BytesIO(response.content))
        return image
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image from URL: {e}")
    except Exception as e:
        raise ValueError(f"Invalid image format or corrupted image: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "ChangeClothesAI service is running"})

@app.route('/model-status', methods=['GET'])
def model_status_check():
    """Check the current status of model loading"""
    global model_status
    
    # Try to trigger loading if not started yet
    if model_status == "not_loaded":
        try:
            get_service()  # This will trigger loading
        except Exception as e:
            return jsonify({
                "model_status": "not_loaded",
                "message": f"Failed to start model loading: {str(e)}",
                "error": True
            })
    
    return jsonify({
        "model_status": model_status,
        "message": f"Models are {model_status}",
        "error": model_status != "loaded"
    })

@app.route('/try-on', methods=['POST'])
def try_on():
    """Main try-on endpoint"""
    try:
        # Check model status first
        global model_status
        
        # Try to get service (this will trigger loading if needed)
        service_instance = get_service()
        
        # If service is None, it means models are loading
        if service_instance is None:
            return jsonify({
                "error": True,
                "model_status": model_status,
                "message": "Models are currently loading. Please check /model-status endpoint and retry when ready.",
                "retry_after": 5
            })
        
        # Get form data
        human_image = request.files.get('human_image')
        garment_image = request.files.get('garment_image')
        garment_description = request.form.get('garment_description', 'a stylish garment')
        category = request.form.get('category', 'upper_body')
        denoise_steps = int(request.form.get('denoise_steps', 30))
        seed = request.form.get('seed')
        if seed and seed != '-1':
            seed = int(seed)
        else:
            seed = None
        auto_mask = request.form.get('auto_mask', 'true').lower() == 'true'
        auto_crop = request.form.get('auto_crop', 'false').lower() == 'true'
        
        # Validate inputs
        if not human_image or not garment_image:
            return jsonify({"error": "Both human_image and garment_image are required"}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images
            human_path = os.path.join(temp_dir, "human.jpg")
            garment_path = os.path.join(temp_dir, "garment.jpg")
            
            human_image.save(human_path)
            garment_image.save(garment_path)
            
            # Run try-on
            result_img, result_mask = service_instance.try_on(
                human_img_path=human_path,
                garment_img_path=garment_path,
                garment_description=garment_description,
                category=category,
                denoise_steps=denoise_steps,
                seed=seed,
                auto_mask=auto_mask,
                auto_crop=auto_crop,
                save_output=False,  # Don't save to disk
                output_path=temp_dir
            )
            
            # Convert images to base64 for response
            def image_to_base64(img):
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode()
            
            # Return results
            return jsonify({
                "success": True,
                "generated_image": image_to_base64(result_img),
                "mask_image": image_to_base64(result_mask),
                "message": "Try-on completed successfully"
            })
            
    except Exception as e:
        print(f"Error in try-on endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/try-on-file', methods=['POST'])
def try_on_file():
    """Alternative endpoint that returns files instead of base64"""
    try:
        # Check model status first
        global model_status
        
        # Try to get service (this will trigger loading if needed)
        service_instance = get_service()
        
        # If service is None, it means models are loading
        if service_instance is None:
            return jsonify({
                "error": True,
                "model_status": model_status,
                "message": "Models are currently loading. Please check /model-status endpoint and retry when ready.",
                "retry_after": 5
            })
        
        # Get form data
        human_image = request.files.get('human_image')
        garment_image = request.files.get('garment_image')
        garment_description = request.form.get('garment_description', 'a stylish garment')
        category = request.form.get('category', 'upper_body')
        denoise_steps = int(request.form.get('denoise_steps', 30))
        seed = request.form.get('seed')
        if seed and seed != '-1':
            seed = int(seed)
        else:
            seed = None
        auto_mask = request.form.get('auto_mask', 'true').lower() == 'true'
        auto_crop = request.form.get('auto_crop', 'false').lower() == 'true'
        
        # Validate inputs
        if not human_image or not garment_image:
            return jsonify({"error": "Both human_image and garment_image are required"}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images
            human_path = os.path.join(temp_dir, "human.jpg")
            garment_path = os.path.join(temp_dir, "garment.jpg")
            
            human_image.save(human_path)
            garment_image.save(garment_path)
            
            # Run try-on
            result_img, result_mask = service_instance.try_on(
                human_img_path=human_path,
                garment_img_path=garment_path,
                garment_description=garment_description,
                category=category,
                denoise_steps=denoise_steps,
                seed=seed,
                auto_mask=auto_mask,
                auto_crop=auto_crop,
                save_output=True,
                output_path=temp_dir
            )
            
            # Return the generated image file
            output_path = os.path.join(temp_dir, "generated_image.png")
            return send_file(output_path, mimetype='image/png')
            
    except Exception as e:
        print(f"Error in try-on-file endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/try-on-url', methods=['POST'])
def try_on_url():
    """Try-on endpoint that accepts image URLs instead of file uploads"""
    try:
        # Check model status first
        global model_status
        
        # Try to get service (this will trigger loading if needed)
        service_instance = get_service()
        
        # If service is None, it means models are loading
        if service_instance is None:
            return jsonify({
                "error": True,
                "model_status": model_status,
                "message": "Models are currently loading. Please check /model-status endpoint and retry when ready.",
                "retry_after": 5
            })
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data is required"}), 400
        
        # Extract parameters
        human_image_url = data.get('human_image_url')
        garment_image_url = data.get('garment_image_url')
        garment_description = data.get('garment_description', 'a stylish garment')
        category = data.get('category', 'upper_body')
        denoise_steps = int(data.get('denoise_steps', 30))
        seed = data.get('seed')
        if seed and seed != -1:
            seed = int(seed)
        else:
            seed = None
        auto_mask = data.get('auto_mask', True)
        auto_crop = data.get('auto_crop', False)
        
        # Validate inputs
        if not human_image_url or not garment_image_url:
            return jsonify({"error": "Both human_image_url and garment_image_url are required"}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download images from URLs
            print(f"Downloading human image from: {human_image_url}")
            human_image = download_image_from_url(human_image_url)
            
            print(f"Downloading garment image from: {garment_image_url}")
            garment_image = download_image_from_url(garment_image_url)
            
            # Save images to temporary files
            human_path = os.path.join(temp_dir, "human.jpg")
            garment_path = os.path.join(temp_dir, "garment.jpg")
            
            human_image.save(human_path)
            garment_image.save(garment_path)
            
            # Run try-on
            result_img, result_mask = service_instance.try_on(
                human_img_path=human_path,
                garment_img_path=garment_path,
                garment_description=garment_description,
                category=category,
                denoise_steps=denoise_steps,
                seed=seed,
                auto_mask=auto_mask,
                auto_crop=auto_crop,
                save_output=False,  # Don't save to disk
                output_path=temp_dir
            )
            
            # Convert images to base64 for response
            def image_to_base64(img):
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode()
            
            # Return results
            return jsonify({
                "success": True,
                "generated_image": image_to_base64(result_img),
                "mask_image": image_to_base64(result_mask),
                "message": "Try-on completed successfully from URLs"
            })
            
    except Exception as e:
        print(f"Error in try-on-url endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/try-on-url-file', methods=['POST'])
def try_on_url_file():
    """Try-on endpoint with URLs that returns the generated image file directly"""
    try:
        # Check model status first
        global model_status
        
        # Try to get service (this will trigger loading if needed)
        service_instance = get_service()
        
        # If service is None, it means models are loading
        if service_instance is None:
            return jsonify({
                "error": True,
                "model_status": model_status,
                "message": "Models are currently loading. Please check /model-status endpoint and retry when ready.",
                "retry_after": 5
            })
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data is required"}), 400
        
        # Extract parameters
        human_image_url = data.get('human_image_url')
        garment_image_url = data.get('garment_image_url')
        garment_description = data.get('garment_description', 'a stylish garment')
        category = data.get('category', 'upper_body')
        denoise_steps = int(data.get('denoise_steps', 30))
        seed = data.get('seed')
        if seed and seed != -1:
            seed = int(seed)
        else:
            seed = None
        auto_mask = data.get('auto_mask', True)
        auto_crop = data.get('auto_crop', False)
        
        # Validate inputs
        if not human_image_url or not garment_image_url:
            return jsonify({"error": "Both human_image_url and garment_image_url are required"}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download images from URLs
            print(f"Downloading human image from: {human_image_url}")
            human_image = download_image_from_url(human_image_url)
            
            print(f"Downloading garment image from: {garment_image_url}")
            garment_image = download_image_from_url(garment_image_url)
            
            # Save images to temporary files
            human_path = os.path.join(temp_dir, "human.jpg")
            garment_path = os.path.join(temp_dir, "garment.jpg")
            
            human_image.save(human_path)
            garment_image.save(garment_path)
            
            # Run try-on
            result_img, result_mask = service_instance.try_on(
                human_img_path=human_path,
                garment_img_path=garment_path,
                garment_description=garment_description,
                category=category,
                denoise_steps=denoise_steps,
                seed=seed,
                auto_mask=auto_mask,
                auto_crop=auto_crop,
                save_output=True,
                output_path=temp_dir
            )
            
            # Return the generated image file
            output_path = os.path.join(temp_dir, "generated_image.png")
            return send_file(output_path, mimetype='image/png')
            
    except Exception as e:
        print(f"Error in try-on-url-file endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def preload_models():
    """Preload models in a background thread"""
    try:
        print("=" * 60)
        print("🔄 Starting model preloading in background...")
        print("=" * 60)
        get_service()
        print("=" * 60)
        print("✅ Models loaded successfully! API is ready to use.")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"❌ Failed to preload models: {e}")
        print("Models will be loaded on first request.")
        print("=" * 60)

@app.after_serving
def after_serving():
    """Start preloading models after server starts but before first request"""
    def preload_in_background():
        preload_models()
    
    preload_thread = threading.Thread(target=preload_in_background, daemon=True)
    preload_thread.start()

if __name__ == '__main__':
    print("Starting ChangeClothesAI API service...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /model-status - Check model loading status")
    print("  POST /try-on - Try-on with file uploads (base64 response)")
    print("  POST /try-on-file - Try-on with file uploads (file response)")
    print("  POST /try-on-url - Try-on with image URLs (base64 response)")
    print("  POST /try-on-url-file - Try-on with image URLs (file response)")
    print("\nTo check model status:")
    print("  curl http://localhost:8000/model-status")
    print("\nTo test with file uploads:")
    print("  curl -X POST -F 'human_image=@human.jpg' -F 'garment_image=@garment.jpg' http://localhost:8000/try-on")
    print("\nTo test with URLs:")
    print('  curl -X POST -H "Content-Type: application/json" -d \'{"human_image_url":"https://example.com/human.jpg","garment_image_url":"https://example.com/garment.jpg"}\' http://localhost:8000/try-on-url')
    print()
    
    print("🚀 Starting Flask server...")
    print("💡 Models will start loading when first request arrives")
    
    app.run(host='0.0.0.0', port=8000, debug=True)