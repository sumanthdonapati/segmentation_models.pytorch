import tempfile
import os
import gc

import numpy as np
import torch
from diffusers.utils import export_to_video, load_image

from pruna_pro import PrunaProModel


class Predictor:
    def setup(self):
        import logging

        logging.basicConfig(level=logging.INFO)
        
        # Load tokens from environment variables
        pruna_token = os.getenv("PRUNA_TOKEN")
        hf_token = os.getenv("PRUNA_HF_TOKEN")
        
        if not pruna_token:
            raise ValueError("PRUNA_TOKEN environment variable is required")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        
        self.pipe = PrunaProModel.from_pretrained(
            "PrunaAI/Wan2.2-I2V-A14B",
            token=pruna_token,
            hf_token=hf_token,
            verbose=True,  # If supported by PrunaProModel
            log_level="info",  # If supported
        )
        self.pipe.transformer.forward = torch.compile(
            self.pipe.transformer.forward, fullgraph=False
        )
        self.pipe.transformer_2.forward = torch.compile(
            self.pipe.transformer_2.forward, fullgraph=False
        )

    def cleanup_gpu_memory(self):
        """Aggressively clean up GPU memory after inference to prevent OOM errors."""
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
            print(f"GPU memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    def predict(
        self,
        prompt,
        image,
        num_frames=81,
        resolution="480p",
        frames_per_second=16,
        go_fast=True,
        seed=None,
        negative_prompt=" "
    ):
        # Log memory usage at start
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"GPU memory at start - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        generator = (
            torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        )
        if resolution == "480p":
            width, height = 480, 832
        else:
            width, height = 720, 1280
        
        # Load image from input.jpg (saved by the handler)
        try:
            image = load_image("input.png")
        except Exception as e:
            raise ValueError(f"Failed to load input.jpg: {str(e)}")
        
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = (
            self.pipe.vae_scale_factor_spatial
            * self.pipe.transformer.config.patch_size[1]
        )
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        if go_fast:
            num_inference_steps = 4
        else:
            num_inference_steps = 8
        with torch.inference_mode(), torch.no_grad():
            output_video = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=1.0,
                guidance_scale_2=1.0,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).frames[0]
        
        output_dir = tempfile.mkdtemp()
        export_to_video(output_video, "output.mp4", fps=frames_per_second)
        
        # Explicitly delete variables to free memory
        del output_video
        if generator is not None:
            del generator
        del image
        
        # Clean up GPU memory after inference
        self.cleanup_gpu_memory()
        
        return "output.mp4"


# Example usage (only runs if file is executed directly)
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    output = predictor.predict(
        prompt="A white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        image="input.jpg",
    )
    print(output)
