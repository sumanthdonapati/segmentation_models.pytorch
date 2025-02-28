from workflow_combined_v5 import CombinedPipelineV5
from segmind_utils import *
from utilities import upload_to_s3, _BUCKET_NAME
import time
import traceback
import runpod
import os
import torch
from PIL import Image
import requests
import json

# Initialize generators
generator = CombinedPipelineV5()

def call_clarity_upscaler(image_path, mask_path, request_id="request_id", **kwargs):
    """Call clarity upscaler API endpoint"""
    output_path = None
    try:
        # Verify input files exist
        if not os.path.exists(image_path):
            raise Exception(f"Input image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise Exception(f"Input mask not found: {mask_path}")
            
        # Prepare the request data with files
        files = {
            'image': ('image.png', open(image_path, 'rb'), 'image/png'),
            'mask': ('mask.png', open(mask_path, 'rb'), 'image/png')
        }
        
        data = {
            'request_id': request_id,
            'prompt': kwargs.get('prompt', "masterpiece, best quality, highres"),
            'negative_prompt': kwargs.get('negative_prompt', "(worst quality, low quality, normal quality:2)"),
            'scale_factor': kwargs.get('scale_factor', 2),
            'dynamic': kwargs.get('dynamic', 6),
            'creativity': kwargs.get('creativity', 0.35),
            'resemblance': kwargs.get('resemblance', 0.6),
            'tiling_width': kwargs.get('tiling_width', 112),
            'tiling_height': kwargs.get('tiling_height', 144),
            'sd_model': kwargs.get('sd_model', "juggernaut_reborn.safetensors"),
            'scheduler': kwargs.get('scheduler', "DPM++ 3M SDE Karras"),
            'num_inference_steps': kwargs.get('num_inference_steps', 18),
            'seed': kwargs.get('seed', -1),
            'output_format': kwargs.get('output_format', "png")
        }
        
        print(f"Calling clarity upscaler API with files: {image_path}, {mask_path}")
        
        # Make request to local clarity upscaler API
        response = requests.post(
            'http://localhost:5501/upscale',
            files=files,  # Send as multipart form data
            data=data    # Send other parameters as form data
        )
        
        if response.status_code != 200:
            raise Exception(f"Clarity upscaler API error: {response.text}")
            
        # Get headers
        generation_time = json.loads(response.headers.get('X-generation-time', '0'))
        output_metadata = json.loads(response.headers.get('X-output-metadata', '{}'))
        gpu_info = json.loads(response.headers.get('x-gpu-info', '{}'))
            
        # Create output directory and path
        output_dir = '/segmind/ComfyUI/output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'clarity_output.{kwargs.get("output_format", "png")}')
        
        # Save response image
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        return {
            'output_path': output_path,
            'generation_time': generation_time,
            'metadata': output_metadata,
            'gpu_info': gpu_info
        }
        
    except Exception as e:
        print(f"Error in clarity upscaler: {str(e)}")
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise

@torch.inference_mode()
def combined_pipeline_handler(job):
    start = time.time()
    try:
        request_id = job["input"].get("request_id", "request_id")
        body = job["input"]
        
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} API Request start time:', start)
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} JSON Body:', body)
        
        # Get and process input image
        image = body.get('image')
        if image is None:
            return get_error_response("Input Image not found", 400, type="serverless")
            
        print(f"Received image input: {type(image)}")
        
        # Create directory if it doesn't exist
        os.makedirs('/segmind/ComfyUI/input', exist_ok=True)
        input_path = '/segmind/ComfyUI/input/input.png'
        
        try:
            # Load and resize image using utility functions
            print('Loading input image...')
            temp_img = load_image(image)
            if temp_img is None:
                raise Exception("Failed to load image")
                
            print('Resizing image...')
            source_img = resize_image(temp_img)
            if source_img is None:
                raise Exception("Failed to resize image")
                
            print('Saving image...')
            source_img.save(input_path)
            print(f"Saved input image to: {input_path}")
            
            # Verify the image was saved
            if not os.path.exists(input_path):
                raise Exception(f"Failed to save image to {input_path}")
                
            # Verify image is readable
            try:
                test_img = Image.open(input_path)
                test_img.verify()
                print(f"Verified image at {input_path}, size: {test_img.size}")
            except Exception as e:
                raise Exception(f"Invalid image at {input_path}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing input image: {str(e)}")
            print("Traceback:", str(traceback.format_exc()))
            return get_error_response(f"Invalid input image: {str(e)}", 400, type="serverless")
        
        # Validate parameters
        verify_response = verify_params(
            type="serverless",
            prompt=body.get("prompt"),
            steps=body.get("steps"),
            guidance_scale=body.get("guidance_scale")
        )
        
        if verify_response is not None:
            return verify_response
            
        # Process through combined pipeline
        infer_start = time.time()
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Pre-processing Time:', infer_start - start)
        
        result = generator(**body)
        
        infer_mid = time.time()
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Stage 1 Generation Time:', infer_mid - infer_start)
        
        # Get paths from result
        enhanced_path = result["enhanced_path"]
        mask_path = result["mask_path"]
        comfy_input_path = result["comfy_input_path"]
        
        try:
            # Call clarity upscaler API
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Starting Stage 2 (Clarity Upscaler)')
            
            output_info = call_clarity_upscaler(
                image_path=enhanced_path,
                mask_path=mask_path,
                request_id=request_id,
                prompt=body.get("upscale_prompt"),
                negative_prompt=body.get("upscale_negative_prompt"),
                scale_factor=body.get("scale_factor"),
                dynamic=body.get("dynamic"),
                creativity=body.get("creativity"),
                resemblance=body.get("resemblance"),
                tiling_width=body.get("tiling_width"),
                tiling_height=body.get("tiling_height"),
                sd_model=body.get("sd_model"),
                scheduler=body.get("scheduler"),
                num_inference_steps=body.get("num_inference_steps"),
                seed=body.get("upscale_seed"),
                output_format=body.get("output_format")
            )
            
            # Upload to S3
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Uploading to S3')
            object_key = f"lowes/{request_id}/{os.path.basename(output_info['output_path'])}"
            try:
                upload_to_s3(output_info['output_path'], object_key)
            except Exception as e:
                print("S3 upload traceback:", str(traceback.format_exc()))
                return get_error_response("Failed while uploading output", 400, type="serverless")
                
            image_url = f"https://{_BUCKET_NAME}.s3.amazonaws.com/{object_key}"
            
            # Clean up
            for path in [input_path, enhanced_path, mask_path, comfy_input_path, output_info['output_path']]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error cleaning up {path}: {str(e)}")
                
            infer_time = output_info['generation_time']
            end_time = time.time()
            
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Stage 1 Time:', infer_mid - infer_start)
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Stage 2 Time:', infer_time)
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Total Generation Time:', infer_mid - infer_start + infer_time)
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Total API Time:', end_time - start)
            
            # Get metadata from output_info
            img_height = output_info['metadata'].get('height', 0)
            img_width = output_info['metadata'].get('width', 0)
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Final Output Dimensions: {img_width}x{img_height}')
            
            # Prepare metadata similar to flask_serverless.py
            meta_data = {
                "height": img_height,
                "width": img_width,
                "samples": 1
            }
            
            # Calculate total inference time
            total_infer_time = infer_mid - infer_start + infer_time
            
            response = [{
                'status': "Success",
                'image': image_url,
                'infer_time': total_infer_time,
                'outputs': 1,
                'image_format': body.get('output_format', 'png')
            }, {
                'X-generation-time': total_infer_time,
                'X-output-metadata': meta_data,
                'x-gpu-info': output_info['gpu_info']
            }]
            
            return response
            
        except Exception as e:
            print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Clarity upscaler error: {str(e)}')
            raise
            
    except Exception as e:
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Error: {str(e)}')
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Traceback:', str(traceback.format_exc()))
        return get_error_response(str(traceback.format_exc()), 500, type="serverless")

print("Running combined serverless pipeline")
runpod.serverless.start({"handler": combined_pipeline_handler}) 
