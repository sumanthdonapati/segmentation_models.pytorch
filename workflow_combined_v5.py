import torch
from workflow_ppv5 import PPV5
from workflow_mask import ComfyText2ImgGenerator
import os
import time
from utilities import upload_to_s3
import sys
import shutil
from PIL import Image

# Create temp directory if it doesn't exist
TEMP_DIR = '/tmp/clarity_upscaler'
os.makedirs(TEMP_DIR, exist_ok=True)

class CombinedPipelineV5:
    def __init__(self):
        self.ppv5_generator = PPV5()
        self.mask_generator = ComfyText2ImgGenerator()
        
        # Create ComfyUI input directory if it doesn't exist
        self.comfy_input_dir = '/segmind/ComfyUI/input'
        os.makedirs(self.comfy_input_dir, exist_ok=True)
        
    @torch.inference_mode()
    def __call__(self, **kwargs):
        enhanced_path = None
        mask_path = None
        comfy_input_path = None
        temp_input_path = None
        
        try:
            # Copy input image to ComfyUI input directory
            input_image = kwargs.get('image')
            comfy_input_path = os.path.join(self.comfy_input_dir, 'input.png')
            temp_input_path = os.path.join(TEMP_DIR, "input.png")
            
            # Download image if it's a URL
            if input_image.startswith('http'):
                import requests
                response = requests.get(input_image)
                # Verify the image was downloaded successfully
                if response.status_code != 200:
                    raise Exception(f"Failed to download image from {input_image}")
                    
                # Save and verify the image
                with open(temp_input_path, 'wb') as f:
                    f.write(response.content)
                # Verify it's a valid image
                try:
                    img = Image.open(temp_input_path)
                    img.verify()
                    print(f"Downloaded image size: {img.size}")
                except Exception as e:
                    raise Exception(f"Invalid image downloaded: {str(e)}")
                    
                shutil.copy2(temp_input_path, comfy_input_path)
            else:
                # Handle base64 image input
                import base64
                import io
                
                try:
                    # Try to decode base64 string
                    im_binary = base64.b64decode(input_image)
                    buf = io.BytesIO(im_binary)
                except:
                    # Handle padding if needed
                    missing_padding = len(input_image) % 4
                    input_image += '=' * (4 - missing_padding)
                    im_binary = base64.b64decode(input_image)
                    buf = io.BytesIO(im_binary)
                
                # Save the decoded image
                img = Image.open(buf)
                img.save(temp_input_path)
                print(f"Base64 decoded image size: {img.size}")
                
                shutil.copy2(temp_input_path, comfy_input_path)
            
            print(f"Input image saved to: {comfy_input_path}")
            
            # Run PPv5 pipeline
            enhanced = self.ppv5_generator(
                prompt=kwargs.get("prompt"),
                lighting_prompt=kwargs.get("lighting_prompt", "natural day lighting, outdoor, patio background"),
                negative_prompt=kwargs.get("negative_prompt", "CGI, Unreal, Airbrushed, Digital, Blur"),
                aspect_ratio=kwargs.get("aspect_ratio", "8:5 (Cinematic View)"),
                seed=kwargs.get("seed", -1),
                detailer_mode=kwargs.get("detailer_mode", "soft_light"),
                x_value=kwargs.get("x_value", 50),
                y_value=kwargs.get("y_value", 65),
                scale=kwargs.get("scale", 0.63),
                steps=kwargs.get("steps", 8),
                guidance_scale=kwargs.get("guidance_scale", 2),
                lighting=kwargs.get("lighting", True),
                detailtransfer=kwargs.get("detailtransfer", True),
                upscale=kwargs.get("upscale", True),
                megapixel=kwargs.get("megapixel", "1.0")
            )
            
            # Save enhanced image temporarily
            enhanced_path = os.path.join(TEMP_DIR, "temp_image.png")
            enhanced.save(enhanced_path)
            
            # Copy enhanced image to ComfyUI input directory
            comfy_input_path = os.path.join(self.comfy_input_dir, 'input-img.png')
            shutil.copy2(enhanced_path, comfy_input_path)
            
            # Stage 2: Mask Generation
            masks = self.mask_generator(
                prompt=kwargs.get("mask_prompt", ""),
                threshold=kwargs.get("threshold", 0.3),
                invert_mask=kwargs.get("invert_mask", False),
                grow_mask=kwargs.get("grow_mask", 0),
                return_mask=True
            )
            
            if not masks:
                raise Exception("Mask generation stage failed")
                
            # Save mask temporarily    
            mask = masks[0]
            mask_path = os.path.join(TEMP_DIR, "temp_mask.png")
            mask.save(mask_path)
            
            return {
                "enhanced_path": enhanced_path,
                "mask_path": mask_path,
                "comfy_input_path": comfy_input_path
            }
            
        except Exception as e:
            # Clean up any temporary files in case of error
            for path in [enhanced_path, mask_path, temp_input_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as cleanup_error:
                        print(f"Error cleaning up {path}: {str(cleanup_error)}")
            raise e 
