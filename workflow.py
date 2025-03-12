import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
import time


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    VAELoader,
    DualCLIPLoader,
    LoraLoaderModelOnly,
    KSampler,
    LoadImage,
    NODE_CLASS_MAPPINGS,
    CLIPVisionLoader,
    InpaintModelConditioning,
    StyleModelLoader,
    UNETLoader,
    EmptyLatentImage,
    EmptyImage,
    ConditioningZeroOut,
    CLIPTextEncode,
    VAEDecode,
)


class FashionWorkflow:
    def __init__(self):
        import_custom_nodes()
        with torch.inference_mode():
            # Initialize all node instances
            self.loadimage = LoadImage()
            self.ttn_text = NODE_CLASS_MAPPINGS["ttN text"]()
            self.df_text = NODE_CLASS_MAPPINGS["DF_Text"]()
            self.cr_text_concatenate = NODE_CLASS_MAPPINGS["CR Text Concatenate"]()
            self.dualcliploader = DualCLIPLoader()
            self.cliptextencode = CLIPTextEncode()
            self.vaeloader = VAELoader()
            self.unetloader = UNETLoader()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.stylemodelloader = StyleModelLoader()
            self.clipvisionloader = CLIPVisionLoader()
            self.emptylatentimage = EmptyLatentImage()
            self.df_get_latent_size = NODE_CLASS_MAPPINGS["DF_Get_latent_size"]()
            self.easy_imagerembg = NODE_CLASS_MAPPINGS["easy imageRemBg"]()
            self.imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
            self.groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything)"]()
            self.sammodelloader_segment_anything = NODE_CLASS_MAPPINGS["SAMModelLoader (segment anything)"]()
            self.groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoSAMSegment (segment anything)"]()
            self.maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
            self.layerutility_imageremovealpha = NODE_CLASS_MAPPINGS["LayerUtility: ImageRemoveAlpha"]()
            self.inpaintcrop = NODE_CLASS_MAPPINGS["InpaintCrop"]()
            self.reduxadvanced = NODE_CLASS_MAPPINGS["ReduxAdvanced"]()
            self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.loraloadermodelonly = LoraLoaderModelOnly()
            self.cr_lora_stack = NODE_CLASS_MAPPINGS["CR LoRA Stack"]()
            self.cr_apply_lora_stack = NODE_CLASS_MAPPINGS["CR Apply LoRA Stack"]()
            self.simplemathint = NODE_CLASS_MAPPINGS["SimpleMathInt+"]()
            self.simplemathfloat = NODE_CLASS_MAPPINGS["SimpleMathFloat+"]()
            self.ksampler = KSampler()
            self.vaedecode = VAEDecode()
            self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            self.any_switch_rgthree = NODE_CLASS_MAPPINGS["Any Switch (rgthree)"]()
            self.sam2modelloader_segment_anything2 = NODE_CLASS_MAPPINGS["SAM2ModelLoader (segment anything2)"]()
            self.groundingdinomodelloader_segment_anything2 = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything2)"]()
            self.groundingdinosam2segment_segment_anything2 = NODE_CLASS_MAPPINGS["GroundingDinoSAM2Segment (segment anything2)"]()
            self.imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
            self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            self.emptyimage = EmptyImage()
            self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
            self.layermask_maskgrow = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()
            self.inpaintmodelconditioning = InpaintModelConditioning()
            self.layerutility_textbox = NODE_CLASS_MAPPINGS["LayerUtility: TextBox"]()
            self.janusmodelloader = NODE_CLASS_MAPPINGS["JanusModelLoader"]()
            self.janusimageunderstanding = NODE_CLASS_MAPPINGS["JanusImageUnderstanding"]()
            self.conditioningzeroout = ConditioningZeroOut()
            self.imageandmaskpreview = NODE_CLASS_MAPPINGS["ImageAndMaskPreview"]()
            self.get_resolution_crystools = NODE_CLASS_MAPPINGS["Get resolution [Crystools]"]()
            self.getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
            self.simplemath = NODE_CLASS_MAPPINGS["SimpleMath+"]()
            self.imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()
            self.inpaintstitch = NODE_CLASS_MAPPINGS["InpaintStitch"]()
            self.getmasksizeandcount = NODE_CLASS_MAPPINGS["GetMaskSizeAndCount"]()
            self.simplecomparison = NODE_CLASS_MAPPINGS["SimpleComparison+"]()
            self.simplecondition = NODE_CLASS_MAPPINGS["SimpleCondition+"]()
            self.impactswitch = NODE_CLASS_MAPPINGS["ImpactSwitch"]()
            
            # Load models
            self.groundingdinomodelloader_segment_anything_10 = self.groundingdinomodelloader_segment_anything.main(
                model_name="GroundingDINO_SwinB (938MB)"
            )
            
            self.sammodelloader_segment_anything_11 = self.sammodelloader_segment_anything.main(
                model_name="sam_hq_vit_h (2.57GB)"
            )
            
            self.dualcliploader_123 = self.dualcliploader.load_clip(
                clip_name1="t5/t5xxl_fp16.safetensors",
                clip_name2="clip_l.safetensors",
                type="flux",
                device="default",
            )
            
            self.vaeloader_121 = self.vaeloader.load_vae(
                vae_name="FLUX1/ae.safetensors"
            )
            
            self.unetloader_122 = self.unetloader.load_unet(
                unet_name="flux1-fill-dev.safetensors", 
                weight_dtype="default"
            )
            
            self.stylemodelloader_163 = self.stylemodelloader.load_style_model(
                style_model_name="flux1-redux-dev.safetensors"
            )
            
            self.clipvisionloader_164 = self.clipvisionloader.load_clip(
                clip_name="sigclip_vision_patch14_384.safetensors"
            )
            
            self.upscalemodelloader_395 = self.upscalemodelloader.load_model(
                model_name="RealESRGAN_x2.pth"
            )
            
            self.unetloader_149 = self.unetloader.load_unet(
                unet_name="flux1-dev-fp8.safetensors", 
                weight_dtype="default"
            )
            
            self.loraloadermodelonly_150 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="flux-dev-alpha-turbo.safetensors",
                strength_model=1,
                model=get_value_at_index(self.unetloader_149, 0),
            )
            
            self.dualcliploader_417 = self.dualcliploader.load_clip(
                clip_name1="long_clip/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors",
                clip_name2="t5/t5xxl_fp16.safetensors",
                type="flux",
                device="default",
            )
            
            self.sam2modelloader_segment_anything2_403 = self.sam2modelloader_segment_anything2.main(
                model_name="sam2_1_hiera_large.pt"
            )
            
            self.groundingdinomodelloader_segment_anything2_405 = self.groundingdinomodelloader_segment_anything2.main(
                model_name="GroundingDINO_SwinB (938MB)"
            )
            
            self.janusmodelloader_289 = self.janusmodelloader.load_model(
                model_name="deepseek-ai/Janus-Pro-7B"
            )
            cr_lora_stack_423 = self.cr_lora_stack.lora_stacker(
                switch_1="On",
                lora_name_1="catvton_flux_lora.safetensors",
                model_weight_1=1,
                clip_weight_1=1,
                switch_2="On",
                lora_name_2="ace_plus/ace_subject.safetensors",
                model_weight_2=0.5,
                clip_weight_2=1,
                switch_3="Off",
                lora_name_3="None",
                model_weight_3=0.6,
                clip_weight_3=0.6,
            )

            self.cr_apply_lora_stack_425 = self.cr_apply_lora_stack.apply_lora_stack(
                model=get_value_at_index(self.unetloader_122, 0),
                clip=get_value_at_index(self.dualcliploader_123, 0),
                lora_stack=get_value_at_index(cr_lora_stack_423, 0),
            )

    def process_image(self, image):
        """Convert tensor image to PIL Image"""
        for res in image:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        return img

    def upscale_image(self, image):
        """Upscale an image using the loaded upscale model"""
        print('upscaling image..')
        imageupscalewithmodel_result = self.imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(self.upscalemodelloader_395, 0),
            image=image
        )
        return self.process_image(imageupscalewithmodel_result)

    def save_image(self, image, filename):
        """Save an image to disk"""
        try:
            img = self.process_image(image)
            img.save(filename)
            print(f"Saved image: {filename}")
            return img
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
            return None

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        # Start timing
        start_time = time.time()
        print("Starting Fashion Workflow...")
        
        # Get parameters from kwargs
        input_image = kwargs.get("input_image", "input_image.png")
        reference_image = kwargs.get("reference_image", None)  # Set to None by default
        reference_mask = kwargs.get("reference_mask", None)  # Set to None by default
        prompt_clothing = kwargs.get("prompt_clothing", "cloth,dress")
        gender = kwargs.get("gender", "woman")
        nationality = kwargs.get("nationality", "indian")
        location = kwargs.get("location", "garden")
        seed = kwargs.get("seed", random.randint(1, 2**64))
        steps = kwargs.get("steps", 25)
        guidance_scale = kwargs.get("guidance_scale", 1)
        upscale = kwargs.get("upscale", False)
        save_images = kwargs.get("save_images", True)
        output_dir = kwargs.get("output_dir", "")
        
        # Load input images
        loadimage_23 = self.loadimage.load_image(image=input_image)
        
        # Process input mask if provided
        if reference_mask:
            print(f"Using provided reference mask: {reference_mask}")
            loadimage_417 = self.loadimage.load_image(image=reference_mask)
            mask_image = get_value_at_index(loadimage_417, 0)
            imagetomask_248 = self.imagetomask.image_to_mask(
                channel="red",
                image=mask_image
            )
            input_mask = get_value_at_index(imagetomask_248, 0)
                
            if save_images:
                self.save_image(input_mask, os.path.join(output_dir, "input_mask.png"))
        
        # Move these definitions outside the else block, before the reference image check
        # Set up text prompts
        ttn_text_278 = self.ttn_text.conmeow(
            text="retain [prompt clothing]. This is a set of images placed side by side, on the left is a [prompt clothing], on the right is a famale model wearing that [prompt clothing]. The [prompt clothing] on the left and the [prompt clothing] on the right are one.\nThe two [prompt clothing]s are completely identical, indistinguishable, they are the same design, same product, same color, same style, same details."
        )

        df_text_281 = self.df_text.get_value(Text=prompt_clothing)

        cr_text_concatenate_437 = get_value_at_index(ttn_text_278, 0).replace("[prompt clothing]", get_value_at_index(df_text_281, 0))

        # Encode text with CLIP
        cliptextencode_120 = self.cliptextencode.encode(
            text=cr_text_concatenate_437,
            clip=get_value_at_index(self.cr_apply_lora_stack_425, 1),
        )

        # Add the missing fluxguidance_130 definition
        cliptextencode_129 = self.cliptextencode.encode(
            text=" ", clip=get_value_at_index(self.cr_apply_lora_stack_425, 1)
        )

        # Now add fluxguidance_130
        fluxguidance_130 = self.fluxguidance.append(
            guidance=50, conditioning=get_value_at_index(cliptextencode_120, 0)
        )

        # Create empty latent image for processing
        emptylatentimage_368 = self.emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        # Get latent size
        df_get_latent_size_385 = self.df_get_latent_size.get_size(
            original=False, latent=get_value_at_index(emptylatentimage_368, 0)
        )

        # Process input image
        easy_imagerembg_428 = self.easy_imagerembg.remove(
            rem_mode="RMBG-2.0",
            image_output="Preview",
            save_prefix="ComfyUI",
            torchscript_jit=False,
            add_background="none",
            images=get_value_at_index(loadimage_23, 0),
        )

        # Resize image
        imageresize_97 = self.imageresize.execute(
            width=get_value_at_index(df_get_latent_size_385, 0),
            height=get_value_at_index(df_get_latent_size_385, 1),
            interpolation="nearest",
            method="pad",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(easy_imagerembg_428, 0),
        )

        # Use provided mask or generate one with segmentation
        if reference_mask:
            # Resize the provided mask to match the input image
            maskblur_413 = self.maskblur.execute(
                amount=6,
                device="auto",
                mask=input_mask,
            )
            loadimage_24 = self.loadimage.load_image(image=reference_image)
            rgba_image = get_value_at_index(loadimage_24, 0)
        else:
            # Segment the image if no mask is provided
            groundingdinosamsegment_segment_anything_96 = self.groundingdinosamsegment_segment_anything.main(
                prompt=prompt_clothing,
                threshold=0.2,
                sam_model=get_value_at_index(self.sammodelloader_segment_anything_11, 0),
                grounding_dino_model=get_value_at_index(
                    self.groundingdinomodelloader_segment_anything_10, 0
                ),
                image=get_value_at_index(imageresize_97, 0),
            )
            # Blur the mask
            maskblur_413 = self.maskblur.execute(
                amount=6,
                device="auto",
                mask=get_value_at_index(groundingdinosamsegment_segment_anything_96, 1),
            )
            rgba_image = get_value_at_index(groundingdinosamsegment_segment_anything_96, 0)

        print("rgba shape:", rgba_image.shape)
        print("mask shape:", get_value_at_index(maskblur_413, 0).shape)
        # Remove alpha from image
        layerutility_imageremovealpha_100 = self.layerutility_imageremovealpha.image_remove_alpha(
            fill_background=True,
            background_color="#ffffff",
            RGBA_image=rgba_image,
            mask=get_value_at_index(maskblur_413, 0),
        )

        # Crop for inpainting
        inpaintcrop_392 = self.inpaintcrop.inpaint_crop(
            context_expand_pixels=20,
            context_expand_factor=1,
            fill_mask_holes=False,
            blur_mask_pixels=16,
            invert_mask=False,
            blend_pixels=16,
            rescale_algorithm="bicubic",
            mode="forced size",
            force_width=get_value_at_index(imageresize_97, 1),
            force_height=get_value_at_index(imageresize_97, 2),
            rescale_factor=1,
            min_width=512,
            min_height=512,
            max_width=768,
            max_height=768,
            padding=32,
            image=get_value_at_index(layerutility_imageremovealpha_100, 0),
            mask=get_value_at_index(maskblur_413, 0),
        )


        # Check if reference image is provided
        if reference_image:
            loadimage_24 = self.loadimage.load_image(image=reference_image)
            reference_img = get_value_at_index(loadimage_24, 0)
            print(f"Using provided reference image: {reference_image}")
        else:
            # If no reference image, we need to generate one
            print("No reference image provided. Generating a model image...")
            
            # Set up LoRA stack
            cr_lora_stack_422 = self.cr_lora_stack.lora_stacker(
                switch_1="On",
                lora_name_1="flux_realism.safetensors",
                model_weight_1=1,
                clip_weight_1=1,
                switch_2="Off",
                lora_name_2="aesthetic_flux.safetensors",
                model_weight_2=0.8,
                clip_weight_2=0.8,
                switch_3="Off",
                lora_name_3="None",
                model_weight_3=0.6,
                clip_weight_3=0.6,
            )

            # Apply LoRA stack
            lr_start_time = time.time()
            cr_apply_lora_stack_419 = self.cr_apply_lora_stack.apply_lora_stack(
                model=get_value_at_index(self.loraloadermodelonly_150, 0),
                clip=get_value_at_index(self.dualcliploader_417, 0),
                lora_stack=get_value_at_index(cr_lora_stack_422, 0),
            )
            lr_end_time = time.time()
            lr_duration = lr_end_time - lr_start_time
            print(f"LoRA stack applied in {lr_duration:.2f} seconds")
            # Set up text for Janus image understanding
            ttn_text_319 = self.ttn_text.conmeow(
                text="Describe a professional [gender] model from [nation] wearing an outfit identical to the one in the given image. Create a suitable scene set in a [location] background, complete with a dynamic pose and complementary accessories. Frame the description as a fashion shot with a cinematic feel, capturing the essence and style of the outfit and the model. The model poses gracefully for a full-body frontal photoshoot, standing in a poised and confident stance wearing silver shining heels."
            )

            # Set up gender, nationality, and location
            layerutility_textbox_438 = self.layerutility_textbox.text_box_node(text=gender)
            layerutility_textbox_439 = self.layerutility_textbox.text_box_node(text=nationality)
            layerutility_textbox_440 = self.layerutility_textbox.text_box_node(text=location)

            # Create final prompt for Janus
            cr_text_concatenate_436 = get_value_at_index(ttn_text_319, 0).replace("[gender]", get_value_at_index(layerutility_textbox_438, 0)).replace("[nation]", get_value_at_index(layerutility_textbox_439, 0)).replace("[location]", get_value_at_index(layerutility_textbox_440, 0))
            
            # Use Janus to analyze the image and generate a description
            janus_start_time = time.time()
            janusimageunderstanding_290 = self.janusimageunderstanding.analyze_image(
                question=cr_text_concatenate_436,
                seed=seed,
                temperature=0.1,
                top_p=1,
                max_new_tokens=100,
                model=get_value_at_index(self.janusmodelloader_289, 0),
                processor=get_value_at_index(self.janusmodelloader_289, 1),
                image=get_value_at_index(inpaintcrop_392, 1),
            )
            janus_end_time = time.time()
            janus_duration = janus_end_time - janus_start_time
            print(f"Janus image understanding completed in {janus_duration:.2f} seconds")
            # Encode detailed prompt
            cliptextencode_424 = self.cliptextencode.encode(
                text=get_value_at_index(janusimageunderstanding_290, 0),
                clip=get_value_at_index(cr_apply_lora_stack_419, 1),
            )

            # Encode negative prompt
            cliptextencode_418 = self.cliptextencode.encode(
                text="blur, compressed, bad quality, low resolution, deformed, mistake, error, wrong, incorrect",
                clip=get_value_at_index(self.dualcliploader_417, 0),
            )

            # Set up dimensions for generation
            simplemathint_244 = self.simplemathint.execute(value=768)
            simplemathint_245 = self.simplemathint.execute(value=1280)

            # Create empty latent for generation
            emptylatentimage_241 = self.emptylatentimage.generate(
                width=get_value_at_index(simplemathint_244, 0),
                height=get_value_at_index(simplemathint_245, 0),
                batch_size=1,
            )

            # Generate the model image
            ksampler_421 = self.ksampler.sample(
                seed=seed,
                steps=steps,
                cfg=guidance_scale,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(cr_apply_lora_stack_419, 0),
                positive=get_value_at_index(cliptextencode_424, 0),
                negative=get_value_at_index(cliptextencode_418, 0),
                latent_image=get_value_at_index(emptylatentimage_241, 0),
            )

            # Decode the generated image
            vaedecode_420 = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_421, 0),
                vae=get_value_at_index(self.vaeloader_121, 0),
            )

            # Upscale if requested
            if upscale:
                imageupscalewithmodel_394 = self.imageupscalewithmodel.upscale(
                    upscale_model=get_value_at_index(self.upscalemodelloader_395, 0),
                    image=get_value_at_index(vaedecode_420, 0),
                )
                reference_img = get_value_at_index(imageupscalewithmodel_394, 0)
            else:
                reference_img = get_value_at_index(vaedecode_420, 0)

        # Segment the clothing in the reference image
        groundingdinosam2segment_segment_anything2_404 = self.groundingdinosam2segment_segment_anything2.main(
            prompt="clothes,dress,strips,lace",
            threshold=0.25,
            sam_model=get_value_at_index(self.sam2modelloader_segment_anything2_403, 0),
            grounding_dino_model=get_value_at_index(
                self.groundingdinomodelloader_segment_anything2_405, 0
            ),
            image=reference_img,
        )

        # Crop for inpainting the reference image
        inpaintcrop_101 = self.inpaintcrop.inpaint_crop(
            context_expand_pixels=20,
            context_expand_factor=1,
            fill_mask_holes=False,
            blur_mask_pixels=16,
            invert_mask=False,
            blend_pixels=16,
            rescale_algorithm="bicubic",
            mode="forced size",
            force_width=get_value_at_index(imageresize_97, 1),
            force_height=get_value_at_index(imageresize_97, 2),
            rescale_factor=1,
            min_width=512,
            min_height=512,
            max_width=768,
            max_height=768,
            padding=32,
            image=reference_img,
            mask=get_value_at_index(groundingdinosam2segment_segment_anything2_404, 1),
        )

        # Fix the reduxadvanced_162 node based on workflow_og.py
        reduxadvanced_162 = self.reduxadvanced.apply_stylemodel(
            downsampling_factor=1,
            downsampling_function="nearest",
            mode="keep aspect ratio",
            weight=1,
            autocrop_margin=0.1,
            conditioning=get_value_at_index(fluxguidance_130, 0),
            style_model=get_value_at_index(self.stylemodelloader_163, 0),
            clip_vision=get_value_at_index(self.clipvisionloader_164, 0),
            image=get_value_at_index(inpaintcrop_392, 1),
        )

        # Add the missing imageconcanate_104 node
        imageconcanate_104 = self.imageconcanate.concatenate(
            direction="left",
            match_image_size=True,
            image1=get_value_at_index(inpaintcrop_101, 1),
            image2=get_value_at_index(inpaintcrop_392, 1),
        )

        # Fix the emptyimage_111 definition
        emptyimage_111 = self.emptyimage.generate(
            width=get_value_at_index(imageresize_97, 1),
            height=get_value_at_index(imageresize_97, 2),
            batch_size=1,
            color=0,
        )

        # Convert mask to image for visualization
        masktoimage_107 = self.masktoimage.mask_to_image(
            mask=get_value_at_index(inpaintcrop_101, 2)
        )

        # Concatenate mask and empty image
        imageconcanate_106 = self.imageconcanate.concatenate(
            direction="left",
            match_image_size=True,
            image1=get_value_at_index(masktoimage_107, 0),
            image2=get_value_at_index(emptyimage_111, 0),
        )

        # Convert concatenated image to mask
        imagetomask_116 = self.imagetomask.image_to_mask(
            channel="red", 
            image=get_value_at_index(imageconcanate_106, 0)
        )

        # Fix the layermask_maskgrow_136 definition to match the original
        layermask_maskgrow_136 = self.layermask_maskgrow.mask_grow(
            invert_mask=False,
            grow=9,
            blur=4,
            mask=get_value_at_index(imagetomask_116, 0),
        )

        # Add the missing inpaintmodelconditioning_132 definition
        inpaintmodelconditioning_132 = self.inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(reduxadvanced_162, 0),
            negative=get_value_at_index(cliptextencode_129, 0),
            vae=get_value_at_index(self.vaeloader_121, 0),
            pixels=get_value_at_index(imageconcanate_104, 0),
            mask=get_value_at_index(layermask_maskgrow_136, 0),
        )

        # Then we can use it in conditioningzeroout_303
        conditioningzeroout_303 = self.conditioningzeroout.zero_out(
            conditioning=get_value_at_index(inpaintmodelconditioning_132, 0)
        )

        # Segment the clothing in the input image
        groundingdinosam2segment_segment_anything2_406 = self.groundingdinosam2segment_segment_anything2.main(
            prompt=get_value_at_index(df_text_281, 0),
            threshold=0.2,
            sam_model=get_value_at_index(
                self.sam2modelloader_segment_anything2_403, 0
            ),
            grounding_dino_model=get_value_at_index(
                self.groundingdinomodelloader_segment_anything2_405, 0
            ),
            image=get_value_at_index(easy_imagerembg_428, 0),
        )

        # Get mask size and count
        getmasksizeandcount_348 = self.getmasksizeandcount.getsize(
            mask=get_value_at_index(
                groundingdinosam2segment_segment_anything2_406, 1
            )
        )

        # Calculate mask ratio
        simplemath_349 = self.simplemath.execute(
            value="b/a",
            a=get_value_at_index(getmasksizeandcount_348, 1),
            b=get_value_at_index(getmasksizeandcount_348, 2),
        )

        # Set up values for comparison
        simplemathfloat_353 = self.simplemathfloat.execute(value=2)
        simplemathfloat_376 = self.simplemathfloat.execute(value=1.2)
        simplemathint_371 = self.simplemathint.execute(value=1)
        simplemathint_379 = self.simplemathint.execute(value=3)
        simplemathint_380 = self.simplemathint.execute(value=2)

        # Compare mask ratio
        simplecomparison_352 = self.simplecomparison.execute(
            comparison="<",
            a=get_value_at_index(simplemath_349, 0),
            b=get_value_at_index(simplemathfloat_353, 0),
        )

        simplecomparison_375 = self.simplecomparison.execute(
            comparison="<",
            a=get_value_at_index(simplemath_349, 1),
            b=get_value_at_index(simplemathfloat_376, 0),
        )

        # Set up conditions based on comparisons
        simplecondition_374 = self.simplecondition.execute(
            evaluate=get_value_at_index(simplecomparison_375, 0),
            on_true=get_value_at_index(simplemathint_379, 0),
            on_false=get_value_at_index(simplemathint_380, 0),
        )

        simplecondition_370 = self.simplecondition.execute(
            evaluate=get_value_at_index(simplecomparison_352, 0),
            on_true=get_value_at_index(simplecondition_374, 0),
            on_false=get_value_at_index(simplemathint_371, 0),
        )

        # Create empty latent for final generation
        emptylatentimage_365 = self.emptylatentimage.generate(
            width=768, height=1280, batch_size=1
        )

        # Switch based on condition
        impactswitch_367 = self.impactswitch.doit(
            select=get_value_at_index(simplecondition_370, 0),
            sel_mode=False,
            input1=get_value_at_index(emptylatentimage_365, 0),
            unique_id=12722165826867722107,
        )

        # Add this before the ksampler_128 node
        # Zero out conditioning for inpainting
        conditioningzeroout_303 = self.conditioningzeroout.zero_out(
            conditioning=get_value_at_index(inpaintmodelconditioning_132, 0)
        )

        # Then the ksampler_128 can use it
        ksampler_128 = self.ksampler.sample(
            seed=seed,
            steps=30,
            cfg=0.98,
            sampler_name="euler_ancestral",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(self.cr_apply_lora_stack_425, 0),
            positive=get_value_at_index(inpaintmodelconditioning_132, 0),
            negative=get_value_at_index(conditioningzeroout_303, 0),
            latent_image=get_value_at_index(inpaintmodelconditioning_132, 2),
        )

        # Decode the final image
        vaedecode_134 = self.vaedecode.decode(
            samples=get_value_at_index(ksampler_128, 0),
            vae=get_value_at_index(self.vaeloader_121, 0),
        )

        # Preview the image with mask
        imageandmaskpreview_170 = self.imageandmaskpreview.execute(
            mask_opacity=0.5,
            mask_color="255, 0, 255",
            pass_through=False,
            image=get_value_at_index(imageconcanate_104, 0),
            mask=get_value_at_index(layermask_maskgrow_136, 0),
        )

        # Get resolution of the cropped images
        get_resolution_crystools_181 = self.get_resolution_crystools.execute(
            image=get_value_at_index(inpaintcrop_101, 1),
            unique_id=1191735164740701907,
        )

        get_resolution_crystools_182 = self.get_resolution_crystools.execute(
            image=get_value_at_index(inpaintcrop_392, 1),
            unique_id=13682970108067554338,
        )

        # Get size of the decoded image
        getimagesize_212 = self.getimagesize.execute(
            image=get_value_at_index(vaedecode_134, 0)
        )

        # Calculate half width
        simplemath_213 = self.simplemath.execute(
            value="a/2", a=get_value_at_index(getimagesize_212, 0)
        )

        # Crop the image
        imagecrop_214 = self.imagecrop.execute(
            width=get_value_at_index(simplemath_213, 0),
            height=get_value_at_index(getimagesize_212, 1),
            position="top-left",
            x_offset=get_value_at_index(simplemath_213, 0),
            y_offset=0,
            image=get_value_at_index(vaedecode_134, 0),
        )

        # Stitch the inpainted image
        inpaintstitch_217 = self.inpaintstitch.inpaint_stitch(
            rescale_algorithm="nearest",
            stitch=get_value_at_index(inpaintcrop_101, 0),
            inpainted_image=get_value_at_index(imagecrop_214, 0),
        )
        
        # Save the stitched image
        if save_images:
            self.save_image(inpaintstitch_217, os.path.join(output_dir, "stitched_image.png"))

        # Final image concatenation
        imageconcanate_221 = self.imageconcanate.concatenate(
            direction="right",
            match_image_size=True,
            image1=get_value_at_index(easy_imagerembg_428, 0),
            image2=get_value_at_index(inpaintstitch_217, 0),
        )

        # Return the final image
        final_image = get_value_at_index(inpaintstitch_217, 0)
        
        if upscale:
            return self.upscale_image(final_image)
        else:
            return self.process_image(final_image)

        end_time = time.time()
        load_time = end_time - start_time
        print(f"Fashion Workflow completed in {load_time:.2f} seconds")


if __name__ == "__main__":
    workflow = FashionWorkflow()
    # Example with reference image
    # result1 = workflow(
    #     input_image="dress_3.png",
    #     reference_image="Ivory Embroidered Kurta Set.jpeg",  # Provide reference image
    #     prompt_clothing="cloth,dress",
    #     gender="woman",
    #     nationality="indian",
    #     location="garden",
    #     seed=random.randint(1, 2**64),
    #     steps=25,
    #     guidance_scale=1,
    #     upscale=False,
    #     save_images=True,
    #     output_dir="with_reference"
    # )
    # result1.save('fashion_output_with_reference.png')
    
    # Example without reference image (will generate one)
    result2 = workflow(
        input_image="dress_3.png",
        reference_image=None,  # No reference image, will generate one
        prompt_clothing="cloth,dress",
        gender="woman",
        nationality="indian",
        location="garden",
        seed=random.randint(1, 2**64),
        steps=25,
        guidance_scale=1,
        upscale=False,
        save_images=True,
        output_dir="without_reference"
    )
    result2.save('fashion_output_generated.png')
