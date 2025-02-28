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
        self.lowes_generator = PPV5()
        self.mask_generator = ComfyText2ImgGenerator()
        
        # Create ComfyUI input directory if it doesn't exist
        self.comfy_input_dir = '/segmind/ComfyUI/input'
        os.makedirs(self.comfy_input_dir, exist_ok=True)
        
    @torch.inference_mode()
    def __call__(self, **kwargs):
        try:
            # Stage 1: Lowes Product Photography
            # The input image should already be at /segmind/ComfyUI/input/input.png
            input_path = os.path.join(self.comfy_input_dir, "input.png")
            
            # Verify input image exists
            if not os.path.exists(input_path):
                raise Exception(f"Input image not found at {input_path}")
                
            print(f"Using input image from: {input_path}")
            
            enhanced_images = self.lowes_generator(
                prompt=kwargs.get("prompt"),
                base_model=kwargs.get("base_model"),
                negative_prompt=kwargs.get("negative_prompt"),
                megapixel=kwargs.get("megapixel","1.0"),
                aspect_ratio=kwargs.get("aspect_ratio"),
                x_value=kwargs.get("x_value"),
                y_value=kwargs.get("y_value"),
                scale=kwargs.get("scale"),
                steps=kwargs.get("steps"),
                guidance_scale=kwargs.get("guidance_scale"),
                seed=kwargs.get("seed"),
                restore_details=kwargs.get("restore_details"),
                detailer_mode=kwargs.get("detailer_mode"),
                blend_value=kwargs.get("blend_value")
            )
            
            if not enhanced_images:
                raise Exception("Product photography stage failed")
                
            # Save enhanced image for mask generation
            enhanced_image = enhanced_images[0]
            enhanced_path = os.path.join(self.comfy_input_dir, "enhanced_output.png")
            enhanced_image.save(enhanced_path)
            
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
                
            # Save mask    
            mask = masks[0]
            mask_path = os.path.join(self.comfy_input_dir, "mask_output.png")
            mask.save(mask_path)
            
            return {
                "enhanced_path": enhanced_path,
                "mask_path": mask_path,
                "comfy_input_path": input_path
            }
        except Exception as e:
            print(f"Error in CombinedPipeline: {str(e)}")
            raise 
