import sys
sys.path.append('/segmind/segmind-models/')
from segmind_utils import *
from flask import Flask, request, jsonify, make_response, abort, send_file
from flask_cors import CORS
import json
from PIL import Image
import os
import time
import traceback
from clarity_pipeline import Predictor
import io

# Create temp directory if it doesn't exist
TEMP_DIR = '/tmp/clarity_upscaler'
os.makedirs(TEMP_DIR, exist_ok=True)

## GPU metadata
print("---- GPU INFO ----")
print("GPU Endpoint: ",endpoint_id)
print("GPU Instance: ",gpu_instance)
print("GPU IP Address: ",gpu_ip_address)
print("GPU ID: ",gpu_id)
print("GPU Provider: ",gpu_provider)
print("GPU Type: ",gpu_instance_type)

generator = Predictor()
slug = 'clarity-upscale'

app = Flask(__name__)

def get_gpu_info():
    """Get GPU metrics and info"""
    try:
        gpu_metrics = {
            'gpu_ip': gpu_ip_address,
            'gpu_id': gpu_id,
            'gpu_provider': gpu_provider,
            'gpu_type': gpu_instance_type
        }
        return gpu_metrics
    except Exception as e:
        print(f"Error getting GPU info: {str(e)}")
        return {}

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        # Get files from request
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Missing image or mask file'}), 400
            
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        # Get form data parameters
        request_id = request.form.get('request_id', 'request_id')
        prompt = request.form.get('prompt', "masterpiece, best quality, highres")
        negative_prompt = request.form.get('negative_prompt', "(worst quality, low quality, normal quality:2)")
        scale_factor = int(request.form.get('scale_factor', 2))
        dynamic = int(request.form.get('dynamic', 6))
        creativity = float(request.form.get('creativity', 0.35))
        resemblance = float(request.form.get('resemblance', 0.6))
        tiling_width = int(request.form.get('tiling_width', 112))
        tiling_height = int(request.form.get('tiling_height', 144))
        sd_model = request.form.get('sd_model', "juggernaut_reborn.safetensors")
        scheduler = request.form.get('scheduler', "DPM++ 3M SDE Karras")
        num_inference_steps = int(request.form.get('num_inference_steps', 18))
        seed = int(request.form.get('seed', -1))
        output_format = request.form.get('output_format', 'png')
        
        # Save uploaded files to temp directory
        temp_image_path = os.path.join(TEMP_DIR, f'{request_id}_image.png')
        temp_mask_path = os.path.join(TEMP_DIR, f'{request_id}_mask.png')
        temp_output_path = os.path.join(TEMP_DIR, f'{request_id}_output.{output_format}')
        
        image_file.save(temp_image_path)
        mask_file.save(temp_mask_path)
        
        # Load images
        image = Image.open(temp_image_path)
        mask = Image.open(temp_mask_path)
        
        # Start timing
        start_time = time.time()
        
        # Process image
        output_image = generator.predict(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale_factor=scale_factor,
            dynamic=dynamic,
            creativity=creativity,
            resemblance=resemblance,
            tiling_width=tiling_width,
            tiling_height=tiling_height,
            sd_model=sd_model,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        
        # Calculate inference time
        infer_time = time.time() - start_time
        
        # Save output image
        output_image[0].save(temp_output_path)
        
        # Get metadata
        metadata = {
            'height': output_image[0].height,
            'width': output_image[0].width
        }
        
        # Get GPU info
        gpu_info = get_gpu_info()
        
        # Create response with headers
        response = make_response(send_file(temp_output_path, mimetype=f'image/{output_format}'))
        response.headers['X-generation-time'] = json.dumps(infer_time)
        response.headers['X-output-metadata'] = json.dumps(metadata)
        response.headers['x-gpu-info'] = json.dumps(gpu_info)
        
        # Clean up temp files
        for file_path in [temp_image_path, temp_mask_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing temp file {file_path}: {str(e)}")
        
        return response
        
    except Exception as e:
        print(f"Error in upscale endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5501)
