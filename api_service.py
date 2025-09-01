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
from app import ChangeClothesAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global service instance
service = None

def get_service():
    """Get or create the ChangeClothesAI service instance"""
    global service
    if service is None:
        print("Initializing ChangeClothesAI service...")
        service = ChangeClothesAI()
        print("Service initialized!")
    return service

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "ChangeClothesAI service is running"})

@app.route('/try-on', methods=['POST'])
def try_on():
    """Main try-on endpoint"""
    try:
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
            
            # Get service and run try-on
            service = get_service()
            result_img, result_mask = service.try_on(
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
            
            # Get service and run try-on
            service = get_service()
            result_img, result_mask = service.try_on(
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

if __name__ == '__main__':
    print("Starting ChangeClothesAI API service...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /try-on - Try-on with base64 response")
    print("  POST /try-on-file - Try-on with file response")
    print("\nTo test with curl:")
    print("  curl -X POST -F 'human_image=@human.jpg' -F 'garment_image=@garment.jpg' http://localhost:5000/try-on")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
