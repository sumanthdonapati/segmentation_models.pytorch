import sys
import time
from segmind_utils import *
import os
import time
import traceback
import runpod
import torch
import gc
from utilities import *
from workflow import Predictor

## GPU metadata
print("---- GPU INFO ----")
print("GPU Instance: ", gpu_instance)
print("GPU IP Address: ", gpu_ip_address)
print("GPU ID: ", gpu_id)
print("GPU Provider: ", gpu_provider)
print("GPU Type: ", gpu_instance_type)
device = "cuda"

# Initialize the model
print("Loading PrunaPro I2V model...")
start_time = time.time()
predictor = Predictor()
predictor.setup()
load_time = time.time() - start_time
print(f"Predictor load time: {load_time:.2f} seconds")
print("PrunaPro I2V model loaded successfully!")

def cleanup_gpu_memory():
    """Aggressively clean up GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        # Clear torch compilation cache
        torch._dynamo.reset()
        
        # Multiple rounds of cache clearing
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # Force memory release
        torch.cuda.ipc_collect()
        
        # Log memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"Flask GPU memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

@torch.inference_mode()
def segmind_i2v_generation(job):
    start = time.time()
    jsonFile = job["input"]
    body = jsonFile
    request_id = body.get("request_id", "request_id")
    
    # Log memory usage at request start
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"[{request_id}] GPU memory at request start - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    try:
        if "prompt" not in body:
            return get_error_response("Prompt is Mandatory and must be string", 400, type="serverless")
        
        if "image" not in body:
            return get_error_response("Image is Mandatory and must be provided", 400, type="serverless")
        
        prompt = body.get("prompt")
        image_input = body.get("image")
        num_frames = int(body.get("num_frames", 81))
        resolution = body.get("resolution", "480p")
        frames_per_second = int(body.get("frames_per_second", 16))
        go_fast = body.get("go_fast", True)
        seed = body.get("seed")
        negative_prompt = body.get("negative_prompt", "")
        
        # Validate parameters
        if not 81 <= num_frames <= 100:
            return get_error_response("Number of frames should be between 81 and 100", 400, type="serverless")
            
        if resolution not in ["480p", "720p"]:
            return get_error_response("Resolution should be either '480p' or '720p'", 400, type="serverless")
            
        if not 5 <= frames_per_second <= 24:
            return get_error_response("Frames per second should be between 5 and 24", 400, type="serverless")
        
        # Load and save image using segmind_utils.load_image directly
        print(f"Loading image from: {image_input}")
        try:
            image = load_image(image_input)
            image.save("input.png")
            print("Image loaded and saved successfully as input.png")
        except Exception as e:
            return get_error_response(f"Invalid image URL or path. Please provide a valid image. Error: {str(e)}", 400, type="serverless")
        
        infer_start = time.time()
        
        # Generate video using the Predictor class
        try:
            video_path = predictor.predict(
                prompt=prompt,
                image="input.png",
                num_frames=num_frames,
                resolution=resolution,
                frames_per_second=frames_per_second,
                go_fast=go_fast,
                seed=seed,
                negative_prompt=negative_prompt
            )
        except Exception as e:
            return get_error_response(f"Video generation failed: {str(e)}", 500, type="serverless")
        
        infer_end = time.time()
        infer_time = [infer_end - infer_start]
        
        #print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} JSON Body:{body}')
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} API Request start time:', start)
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Pre-processing Time:{infer_start-start}')
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Generation Time:{infer_time[0]}')
        
        # Get video dimensions from the predictor's last processed image
        if resolution == "480p":
            width, height = 480, 832
        else:
            width, height = 720, 1280
        
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Video dimensions: {width}x{height}, frames: {num_frames}')
        
        # Upload video to S3 and return URL
        slug = "wan_gen_i2v"
        object_key = f"{slug}/{request_id}/output.mp4"
        
        try:
            upload_to_s3(video_path, object_key)
            video_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
        except Exception as e:
            print("S3 upload traceback:", str(traceback.format_exc()))
            return get_error_response("Failed while returning the output", 400, type="serverless")
        finally:
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists("input.png"):
                os.remove("input.png")
            # Additional GPU memory cleanup to ensure memory is freed even on errors
            cleanup_gpu_memory()
        
        meta_data = {"height": height, "width": width, "frames": num_frames, "fps": frames_per_second, "resolution": resolution}
        headers = {"X-generation-time": infer_time[0], "X-output-metadata": meta_data, "x-gpu-info": gpu_metrics}
        
        response = [{'status': "Success", 'video': video_url, 'infer_time': infer_time, 'outputs': 1, 'video_format': 'mp4'}, headers]
        
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Post-processing Time:{time.time()-infer_end}')
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} API Response Time:{time.time()}')
        
        return response
        
    except Exception as e:
        print(f'{[request_id]} {[gpu_id]} {[endpoint_id]} Error:{e}')
        print("trace back:", str(traceback.format_exc()))
        # Clean up GPU memory on error
        cleanup_gpu_memory()
        if "invalid literal" in str(e):
            return get_error_response("Inputs which are supposed to be numbers are in alphabet", 400, type="serverless")
        else:
            return get_error_response(str(traceback.format_exc()), 500, type="serverless")

print("running serverless")
runpod.serverless.start({"handler": segmind_i2v_generation})
