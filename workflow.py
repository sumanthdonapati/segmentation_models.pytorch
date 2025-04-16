import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
import time

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    CLIPVisionEncode,
    EmptyImage,
    UNETLoader,
    StyleModelApply,
    DualCLIPLoader,
    InpaintModelConditioning,
    ControlNetApplyAdvanced,
    StyleModelLoader,
    CLIPTextEncode,
    VAEDecode,
    VAELoader,
    LoraLoader,
    ControlNetLoader,
    LoadImage,
    NODE_CLASS_MAPPINGS,
    EmptyLatentImage,
)


class SegfitCore:
    def __init__(self):
        import_custom_nodes()
        self.groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything)"]()
        self.sammodelloader_segment_anything = NODE_CLASS_MAPPINGS["SAMModelLoader (segment anything)"]()
        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.ttn_text = NODE_CLASS_MAPPINGS["ttN text"]()
        self.df_text = NODE_CLASS_MAPPINGS["DF_Text"]()
        self.rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
        self.unetloader = UNETLoader()
        self.dualcliploader = DualCLIPLoader()
        self.loadimage = LoadImage()
        self.cliptextencode = CLIPTextEncode()
        self.vaeloader = VAELoader()
        self.loraloader = LoraLoader()
        self.clipvisionencode = CLIPVisionEncode()
        self.stylemodelloader = StyleModelLoader()
        
        # Initialize Gemini node
        self.gemininode = NODE_CLASS_MAPPINGS["GeminiNode"]()
        
        # Initialize additional nodes that were previously in __call__
        self.emptylatentimage = EmptyLatentImage()
        self.getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
        self.df_get_latent_size = NODE_CLASS_MAPPINGS["DF_Get_latent_size"]()
        self.imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        self.resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
        
        # Additional nodes from __call__
        self.differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        self.basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        self.basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        self.samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        self.vaedecode = VAEDecode()
        self.simplemath = NODE_CLASS_MAPPINGS["SimpleMath+"]()
        self.imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()
        self.inpaintstitchimproved = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()
        self.image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
        self.clothessegment = NODE_CLASS_MAPPINGS["ClothesSegment"]()
        self.get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
        self.image_resize = NODE_CLASS_MAPPINGS["Image Resize"]()
        self.maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        self.maskfix = NODE_CLASS_MAPPINGS["MaskFix+"]()
        self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        self.setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        self.aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        self.controlnetapplyadvanced = ControlNetApplyAdvanced()
        self.inpaintmodelconditioning = InpaintModelConditioning()
        self.randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        self.ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.simplemathint = NODE_CLASS_MAPPINGS["SimpleMathInt+"]()
        self.controlnetloader = ControlNetLoader()
        self.imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
        self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        self.emptyimage = EmptyImage()
        self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        self.layermask_maskgrow = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()
        self.inpaintcrop = NODE_CLASS_MAPPINGS["InpaintCrop"]()
        self.constrainimagepysssss = NODE_CLASS_MAPPINGS["ConstrainImage|pysssss"]()
        self.easy_promptreplace = NODE_CLASS_MAPPINGS["easy promptReplace"]()
        self.stylemodelapply = StyleModelApply()
        self.advancedvisionloader = NODE_CLASS_MAPPINGS["AdvancedVisionLoader"]()
        self.groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoSAMSegment (segment anything)"]()
        self.layerutility_imageremovealpha = NODE_CLASS_MAPPINGS["LayerUtility: ImageRemoveAlpha"]()
        
        # Add nodes from automask.py for cloth detection
        self.dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        self.layerutility_llamavision = NODE_CLASS_MAPPINGS["LayerUtility: LlamaVision"]()
        self.images_to_rgb = NODE_CLASS_MAPPINGS["Images to RGB"]()
        self.ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        self.ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        self.solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
        self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        self.sam2modelloader_segment_anything2 = NODE_CLASS_MAPPINGS["SAM2ModelLoader (segment anything2)"]()
        self.groundingdinomodelloader_segment_anything2 = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything2)"]()
        self.groundingdinosam2segment_segment_anything2 = NODE_CLASS_MAPPINGS["GroundingDinoSAM2Segment (segment anything2)"]()
        self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        # Load models that don't depend on runtime inputs
        self.groundingdinomodelloader_segment_anything_10 = self.groundingdinomodelloader_segment_anything.main(model_name="GroundingDINO_SwinB (938MB)")
        self.sammodelloader_segment_anything_11 = self.sammodelloader_segment_anything.main(model_name="sam_hq_vit_h (2.57GB)")
        self.upscalemodelloader_14 = self.upscalemodelloader.load_model(model_name="RealESRGAN_x2.pth")
        # Load models for sam2
        self.sam2modelloader_segment_anything2_8 = self.sam2modelloader_segment_anything2.main(
            model_name="sam2_1_hiera_large.pt"
        )
        self.groundingdinomodelloader_segment_anything2_9 = self.groundingdinomodelloader_segment_anything2.main(
            model_name="GroundingDINO_SwinB (938MB)"
        )

        self.unetloader_122 = self.unetloader.load_unet(
            unet_name="flux1-fill-dev.safetensors", weight_dtype="fp8_e4m3fn"
        )

        self.dualcliploader_123 = self.dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5/t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )

        self.loraloader_606 = self.loraloader.load_lora(
            lora_name="comfyui_subject_lora16.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(self.unetloader_122, 0),
            clip=get_value_at_index(self.dualcliploader_123, 0),
        )

        self.differentialdiffusion_396 = self.differentialdiffusion.apply(
            model=get_value_at_index(self.loraloader_606, 0)
        )
        self.vaeloader_121 = self.vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")
        self.stylemodelloader_163 = self.stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        self.advancedvisionloader_741 = self.advancedvisionloader.load_vision(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )
        
        self.controlnetloader_804 = self.controlnetloader.load_controlnet(
            control_net_name="FLUX.1/Shakker-Labs-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors"
        )
        
        # Load models for cloth detection from automask.py
        self.checkpointloadersimple_32 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernautXL_lightning.safetensors"
        )
        
        self.clipvisionloader_17 = self.clipvisionloader.load_clip(
            clip_name="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        )
        
        self.controlnetloader_34 = self.controlnetloader.load_controlnet(
            control_net_name="SDXL/OpenPoseXL2.safetensors"
        )
        
        # Add nodes from model_gen.py
        self.cr_lora_stack = NODE_CLASS_MAPPINGS["CR LoRA Stack"]()
        self.cr_apply_lora_stack = NODE_CLASS_MAPPINGS["CR Apply LoRA Stack"]()
        self.showtextpysssss = NODE_CLASS_MAPPINGS["ShowText|pysssss"]()
        
        # Load models for model generation
        self.unetloader_33 = self.unetloader.load_unet(
            unet_name="FLUX1/flux1-dev-fp8.safetensors", weight_dtype="fp8_e5m2"
        )
        self.cr_lora_stack_39 = self.cr_lora_stack.lora_stacker(
            switch_1="On",
            lora_name_1="hands.safetensors",
            model_weight_1=0.8,
            clip_weight_1=0.8,
            switch_2="Off",
            lora_name_2="None",
            model_weight_2=1,
            clip_weight_2=1,
            switch_3="Off",
            lora_name_3="None",
            model_weight_3=1,
            clip_weight_3=1,
        )
        self.cr_lora_stack_26 = self.cr_lora_stack.lora_stacker(
            switch_1="On",
            lora_name_1="flux_realism_lora.safetensors",
            model_weight_1=1,
            clip_weight_1=1,
            switch_2="On",
            lora_name_2="aestheticv5.safetensors",
            model_weight_2=0.8,
            clip_weight_2=0.8,
            switch_3="On",
            lora_name_3="FLUX-dev-lora-AntiBlur.safetensors",
            model_weight_3=1,
            clip_weight_3=1,
            lora_stack=get_value_at_index(self.cr_lora_stack_39, 0),
        )

        self.cr_apply_lora_stack_16 = self.cr_apply_lora_stack.apply_lora_stack(
            model=get_value_at_index(self.unetloader_33, 0),
            clip=get_value_at_index(self.dualcliploader_123, 0),
            lora_stack=get_value_at_index(self.cr_lora_stack_26, 0),
        )
        
    def cloth_detector(self, seed=None):
        """Detect cloth type using Gemini instead of LlamaVision"""
        if seed is None:
            seed = random.randint(1, 2**32)
            
        loadimage_43 = self.loadimage.load_image(image="outfit.png")
        
        rmbg_23 = self.rmbg.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            background="gray",
            invert_output=False,
            optimize="default",
            refine_foreground=False,
            image=get_value_at_index(loadimage_43, 0),
        )
        
        start_time2 = time.time()
        # Replace LlamaVision with Gemini
        gemininode_cloth = self.gemininode.generate_content(
            prompt="You are a professional fashion expert and garment identifier. Your task is to: Analyze the given image and identify the primary clothing item worn or displayed. Based on your identification, return only one of the following labels that best describes the clothing item: 'top', 'bottom', or 'dress' Strict Output Rules: Return the label in lowercase Use a single word only Do not include any explanation, extra words, symbols, or formatting Do not add any punctuation, including no full stop",
            operation_mode="analysis",
            model_name="gemini-2.0-flash",
            temperature=0.1,
            seed=seed,
            sequential_generation=False,
            batch_count=1,
            aspect_ratio="none",
            external_api_key=GEMINI_API_KEY,
            chat_mode=False,
            clear_history=False,
            structured_output=False,
            max_images=6,
            max_output_tokens=10,
            use_random_seed=False,
            api_call_delay=1,
            images=get_value_at_index(rmbg_23, 0),
        )
        end_time2 = time.time()
        print(f"Time taken to generate cloth type: {end_time2 - start_time2} seconds")
        
        # Extract the cloth type from the response
        response = get_value_at_index(gemininode_cloth, 0).lower().strip()
        
        # Check if the response contains any of our target categories
        if "dress" in response:
            return "dress"
        elif "top" in response or "shirt" in response or "blouse" in response:
            return "top"
        elif "bottom" in response or "pants" in response or "skirt" in response:
            return "bottom"
        else:
            return "unknown"
    
    def intermediate_gen(self, cloth_description, seed=None):
        """Generate intermediate image using Gemini instead of LlamaVision"""
        if seed is None:
            seed = random.randint(1, 2**32)
            
        loadimage_43 = self.loadimage.load_image(image="outfit.png")
        loadimage_44 = self.loadimage.load_image(image="model.png")
        
        cliptextencode_4 = self.cliptextencode.encode(
            text="hat,cap,multiple_hands, multiple_legs, multiple_girls\nlow quality, blurry, out of focus, distorted, unrealistic, extra limbs, missing limbs, deformed hands, deformed fingers, extra fingers, long neck, unnatural face, bad anatomy, bad proportions, poorly drawn face, poorly drawn eyes, asymmetrical eyes, extra eyes, extra head, floating objects, watermark, text, logo, jpeg artifacts, overexposed, underexposed, harsh lighting, bad posture, strange angles, unnatural expressions, oversaturated colors, messy hair, unrealistic skin texture, wrinkles inappropriately placed, incorrect shading, pixelation, complex background, busy background, detailed background, crowded scene, clutter, messy elements, unnecessary objects, overlapping objects, intricate patterns, vibrant colors, high contrast, graffiti, shadows, reflections, multiple layers, unrealistic lighting, overexposed areas.",
            clip=get_value_at_index(self.checkpointloadersimple_32, 1),
        )
        
        rmbg_23 = self.rmbg.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            background="gray",
            invert_output=False,
            optimize="default",
            refine_foreground=False,
            image=get_value_at_index(loadimage_43, 0),
        )
        
        start_time3 = time.time()
        # Replace LlamaVision with Gemini
        gemininode_intermediate = self.gemininode.generate_content(
            prompt=f"You are a fashion expert and a prompt engineer, you describe the how clothes fit on a fashion model\n\n- strictly start the prompt with 'A fashion model wearing..' and describe the dress\n\n- describe about the clothes like shirt, pant or dress etc., precisely\n\n -input image contains {cloth_description} cloth/outfit image and want a fashion model wearing it - provide the prompt in paragraph style in less than 200 words",
            operation_mode="analysis",
            model_name="gemini-2.0-flash",
            temperature=0.3,
            seed=seed,
            sequential_generation=False,
            batch_count=1,
            aspect_ratio="none",
            external_api_key=GEMINI_API_KEY,
            chat_mode=False,
            clear_history=False,
            structured_output=False,
            max_images=6,
            max_output_tokens=256,
            use_random_seed=False,
            api_call_delay=1,
            images=get_value_at_index(rmbg_23, 0),
        )
        end_time3 = time.time()
        print(f"Time taken to generate intermediate prompt: {end_time3 - start_time3} seconds")
        
        cliptextencode_12 = self.cliptextencode.encode(
            text=get_value_at_index(gemininode_intermediate, 0),
            clip=get_value_at_index(self.checkpointloadersimple_32, 1),
        )
        
        rmbg_39 = self.rmbg.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            background="gray",
            invert_output=False,
            optimize="default",
            refine_foreground=False,
            image=get_value_at_index(loadimage_44, 0),
        )
        
        constrainimagepysssss_10 = self.constrainimagepysssss.constrain_image(
            max_width=1024,
            max_height=1024,
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(rmbg_39, 0),
        )
        
        dwpreprocessor_42 = self.dwpreprocessor.estimate_pose(
            detect_hand="enable",
            detect_body="enable",
            detect_face="enable",
            resolution=512,
            bbox_detector="yolox_l.onnx",
            pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
            scale_stick_for_xinsr_cn="disable",
            image=get_value_at_index(constrainimagepysssss_10, 0)[0],
        )
        
        imageresizekj_35 = self.imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(dwpreprocessor_42, 0),
            get_image_size=get_value_at_index(constrainimagepysssss_10, 0)[0],
        )
        
        controlnetapplyadvanced_36 = self.controlnetapplyadvanced.apply_controlnet(
            strength=1.5000000000000002,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cliptextencode_12, 0),
            negative=get_value_at_index(cliptextencode_4, 0),
            control_net=get_value_at_index(self.controlnetloader_34, 0),
            image=get_value_at_index(imageresizekj_35, 0),
            vae=get_value_at_index(self.checkpointloadersimple_32, 2),
        )
        
        get_image_size_9 = self.get_image_size.get_size(
            image=get_value_at_index(constrainimagepysssss_10, 0)[0]
        )
        
        solidmask_2 = self.solidmask.solid(
            value=1,
            width=get_value_at_index(get_image_size_9, 0),
            height=get_value_at_index(get_image_size_9, 1),
        )
        
        inpaintmodelconditioning_6 = self.inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(controlnetapplyadvanced_36, 0),
            negative=get_value_at_index(controlnetapplyadvanced_36, 1),
            vae=get_value_at_index(self.checkpointloadersimple_32, 2),
            pixels=get_value_at_index(constrainimagepysssss_10, 0)[0],
            mask=get_value_at_index(solidmask_2, 0),
        )
        
        class_selections = {
            "Hat": False,
            "Hair": False,
            "Face": False,
            "Sunglasses": False,
            "Upper-clothes": True,
            "Skirt": True,
            "Dress": True,
            "Belt": True,
            "Pants": True,
            "Left-arm": False,
            "Right-arm": False,
            "Left-leg": False,
            "Right-leg": False,
            "Bag": False,
            "Scarf": False,
            "Left-shoe": False,
            "Right-shoe": False,
            "Background": False
        }
        
        clothessegment_22 = self.clothessegment.segment_clothes(
            process_res=512,
            mask_blur=0,
            mask_offset=0,
            background_color="gray",
            invert_output=False,
            images=get_value_at_index(rmbg_23, 0),
            **class_selections
        )
        
        images_to_rgb_1 = self.images_to_rgb.image_to_rgb(
            images=get_value_at_index(clothessegment_22, 0)
        )
        
        ipadapterunifiedloader_3 = self.ipadapterunifiedloader.load_models(
            preset="PLUS (high strength)",
            model=get_value_at_index(self.checkpointloadersimple_32, 0),
        )
        
        ipadapteradvanced_30 = self.ipadapteradvanced.apply_ipadapter(
            weight=1.0000000000000002,
            weight_type="style transfer",
            combine_embeds="concat",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_3, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_3, 1),
            image=get_value_at_index(images_to_rgb_1, 0),
            clip_vision=get_value_at_index(self.clipvisionloader_17, 0),
        )
        
        ksampler_13 = self.ksampler.sample(
            seed=seed,
            steps=8,
            cfg=3,
            sampler_name="dpmpp_sde_gpu",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(ipadapteradvanced_30, 0),
            positive=get_value_at_index(inpaintmodelconditioning_6, 0),
            negative=get_value_at_index(inpaintmodelconditioning_6, 1),
            latent_image=get_value_at_index(inpaintmodelconditioning_6, 2),
        )
        
        vaedecode_19 = self.vaedecode.decode(
            samples=get_value_at_index(ksampler_13, 0),
            vae=get_value_at_index(self.checkpointloadersimple_32, 2),
        )
        
        imageresizekj_8 = self.imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(vaedecode_19, 0),
            get_image_size=get_value_at_index(loadimage_44, 0),
        )
        
        # Save the intermediate image
        import numpy as np
        from PIL import Image
        for res in imageresizekj_8[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            img.save("ComfyUI/input/intermediate_image.png")
        
        return img
    
    def model_gen(self, aspect_ratio, cloth_description, model_description, background_description, seed=42):
        """Generate a model image using Gemini instead of LlamaVision"""
        print("Generating model image...")
        aspect_ratio_mapping = {
            "1:1": "1024x1024",
            "2:3": "832x1216",
            "3:4": "896x1152",
            "5:8": "768x1216",
            "9:16": "768x1344",
            "9:19": "704x1472",
            "9:21": "640x1536",
            "3:2": "1216x832",
            "4:3": "1152x896",
            "8:5": "1216x768",
            "16:9": "1344x768",
            "19:9": "1472x704",
            "21:9": "1536x640"
        }
        width, height = aspect_ratio_mapping[aspect_ratio].split('x')
        if aspect_ratio not in aspect_ratio_mapping:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Please use one of the following: {', '.join(aspect_ratio_mapping.keys())}")

        simplemathint_9 = self.simplemathint.execute(value=int(height))
        simplemathint_10 = self.simplemathint.execute(value=int(width))

        loadimage_35 = self.loadimage.load_image(image="outfit.png")
        
        start_time4 = time.time()
        # Replace LlamaVision with Gemini
        gemininode_model = self.gemininode.generate_content(
            prompt=f"You are an expert prompt generator specializing in high-quality, fashion-focused imagery. Your task is to generate a natural, paragraph-style text prompt that describes a fashion model in vivid visual detail. This prompt will be used with AI image generation tools to create photorealistic fashion images. User Inputs May Include: Gender, age, ethnicity, and skin tone Body type or figure Facial features, hairstyle, and expression Outfit or dress type (e.g., saree, crop top and jeans, lehenga, trench coat, gown) Clothing fit (e.g., body-hugging, flowy, structured, tailored, oversized, draped) Clothing length if applicable (e.g., knee-length, ankle-length, floor-length, mini, cropped, full-sleeved) Model's pose or stance Camera angle or framing (e.g., full-body, side view, portrait) Background or setting (optional) Lighting style or mood (optional), the image contains {cloth_description} cloth/outfit image and want a fashion model {model_description} wearing it with good {background_description} background Output Rules: Return a single paragraph that reads smoothly and naturally, suitable for direct use in AI image generation The prompt must describe the type of dress, how it fits on the model, and include the dress length when visually relevant (e.g., 'a knee-length fitted dress that hugs the waist') Use elegant, fashion-forward language (e.g., 'studio-lit editorial portrait', 'confident runway stance', etc.) Do not include lists, bullet points, line breaks, technical syntax, or parameter codes Do not include negative prompts or extra explanations—only the final generated paragraph",
            operation_mode="analysis",
            model_name="gemini-2.0-flash",
            temperature=0.3,
            seed=seed,
            sequential_generation=False,
            batch_count=1,
            aspect_ratio="none",
            external_api_key=GEMINI_API_KEY,
            chat_mode=False,
            clear_history=False,
            structured_output=False,
            max_images=6,
            max_output_tokens=256,
            use_random_seed=False,
            api_call_delay=1,
            images=get_value_at_index(loadimage_35, 0),
        )
        end_time4 = time.time()
        print(f"Time taken to generate model prompt: {end_time4 - start_time4} seconds")
        
        # Get model description from Gemini
        model_prompt = get_value_at_index(gemininode_model, 0)
        print("Model prompt:", model_prompt)
        # Encode text for generation
        cliptextencode_14 = self.cliptextencode.encode(
            text=model_prompt,
            clip=get_value_at_index(self.cr_apply_lora_stack_16, 1),
        )

        cliptextencode_15 = self.cliptextencode.encode(
            text="blur, compressed, bad quality, low resolution, deformed, mistake, error, wrong, incorrect",
            clip=get_value_at_index(self.dualcliploader_123, 0),
        )
    
        # Apply flux guidance
        fluxguidance_8 = self.fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_14, 0)
        )

        # Generate empty latent image
        emptylatentimage_23 = self.emptylatentimage.generate(
            width=get_value_at_index(simplemathint_10, 0),
            height=get_value_at_index(simplemathint_9, 0),
            batch_size=1,
        )

        # Sample the image
        ksampler_24 = self.ksampler.sample(
            seed=seed,
            steps=25,
            cfg=1,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(self.cr_apply_lora_stack_16, 0),
            positive=get_value_at_index(fluxguidance_8, 0),
            negative=get_value_at_index(cliptextencode_15, 0),
            latent_image=get_value_at_index(emptylatentimage_23, 0),
        )

        # Decode the image
        vaedecode_30 = self.vaedecode.decode(
            samples=get_value_at_index(ksampler_24, 0),
            vae=get_value_at_index(self.vaeloader_121, 0),
        )

        # Save the generated model image
        for res in vaedecode_30[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            img.save("ComfyUI/input/model.png")
            
        print("Model image generated and saved as 'ComfyUI/input/model.png'")
        return img

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        full_start_time = time.time()
        seed = kwargs.get("seed", random.randint(1, 2**32))
        upscale = kwargs.get("upscale", False)
        controlnet = kwargs.get("controlnet", None)
        cn_strength = kwargs.get("cn_strength", 0.3)
        cn_start = kwargs.get("cn_start", 0.3)
        cn_end = kwargs.get("cn_end", 1)
        cloth_description = kwargs.get("cloth_description", "")
        model_description = kwargs.get("model_description", "")
        background_description = kwargs.get("background_description", "")
        aspect_ratio = kwargs.get("aspect_ratio", "2:3")
        model_type = kwargs.get("model_type", "quality")
        
        load_image_start = time.time()
        self.loadimage_23 = self.loadimage.load_image(image="outfit.png")
        load_image_end = time.time()
        print(f"Time taken to load outfit image: {load_image_end - load_image_start} seconds")

        if not kwargs.get("model_image"):
            print("No model image detected, generating model image")
            model_gen_start = time.time()
            self.model_gen(aspect_ratio, cloth_description, model_description, background_description, seed=seed)
            model_gen_end = time.time()
            print(f"Time taken to generate model image: {model_gen_end - model_gen_start} seconds")
            
        load_model_image_start = time.time()
        self.loadimage_24 = self.loadimage.load_image(image="model.png")
        get_image_size_17 = self.get_image_size.get_size(
                image=get_value_at_index(self.loadimage_24, 0)
            )
        load_model_image_end = time.time()
        print(f"Time taken to load model image: {load_model_image_end - load_model_image_start} seconds")
        
        og_width, og_height = get_image_size_17[0], get_image_size_17[1]
        if og_width < 768 or og_height < 768:
            print("Model image is too small, upscaling...")
            start_time5 = time.time()
            imageupscalewithmodel_15 = self.imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(self.upscalemodelloader_14, 0),
                image=get_value_at_index(self.loadimage_24, 0),
            )
            end_time5 = time.time()
            print(f"Time taken to upscale model image: {end_time5 - start_time5} seconds")
        else:
            imageupscalewithmodel_15 = self.loadimage_24
            
        if model_type == "quality":
            img_size = 1536
        elif model_type == "balanced":
            img_size = 1280
        else:
            img_size = 1024
            
        constrain_image_start = time.time()
        constrainimagepysssss_795 = self.constrainimagepysssss.constrain_image(
            max_width=img_size,
            max_height=img_size,
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(imageupscalewithmodel_15, 0),
        )
        constrain_image_end = time.time()
        print(f"Time taken to constrain image: {constrain_image_end - constrain_image_start} seconds")

        mask_processing_start = time.time()
        if kwargs.get("model_mask"):
            self.loadimage_25 = self.loadimage.load_image(image="model_mask.png")
            imagetomask_999 = self.imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(self.loadimage_25, 0)
            )
            intermediate_mask = get_value_at_index(imagetomask_999, 0)
        else:
            print("No model mask detected, generating intermediate image")
            intermediate_gen_start = time.time()
            self.intermediate_gen(cloth_description, seed=seed)
            intermediate_gen_end = time.time()
            print(f"Time taken to generate intermediate image: {intermediate_gen_end - intermediate_gen_start} seconds")
            
            cloth_detect_start = time.time()
            cloth_type = self.cloth_detector(seed=seed)
            cloth_detect_end = time.time()
            print(f"Time taken to detect cloth type: {cloth_detect_end - cloth_detect_start} seconds")
            
            print("cloth_type: ", cloth_type)
            if cloth_type not in ["dress", "bottom", "top"]:
                cloth_type = self.cloth_detector()
                if cloth_type not in ["dress", "bottom", "top"]:
                    raise ValueError("Unable to detect cloth type. Please try again.")
            if cloth_type == "dress":
                class_selections = {"Hat": False,"Hair": False,"Face": False,"Sunglasses": False,"Upper-clothes": True,"Skirt": True,"Dress": True,"Belt": True,"Pants": True,"Left-arm": False,"Right-arm": False,"Left-leg": False,"Right-leg": False,"Bag": False,"Scarf": False,"Left-shoe": False,"Right-shoe": False,"Background": False}
            elif cloth_type == "bottom":
                class_selections = {"Hat": False,"Hair": False,"Face": False,"Sunglasses": False,"Upper-clothes": False,"Skirt": True,"Dress": False,"Belt": True,"Pants": True,"Left-arm": False,"Right-arm": False,"Left-leg": False,"Right-leg": False,"Bag": False,"Scarf": False,"Left-shoe": False,"Right-shoe": False,"Background": False}
            elif cloth_type == "top":
                class_selections = {"Hat": False,"Hair": False,"Face": False,"Sunglasses": False,"Upper-clothes": True,"Skirt": False,"Dress": False,"Belt": False,"Pants": False,"Left-arm": False,"Right-arm": False,"Left-leg": False,"Right-leg": False,"Bag": False,"Scarf": False,"Left-shoe": False,"Right-shoe": False,"Background": False}
            
            get_image_size_start = time.time()
            get_image_size_813 = self.get_image_size.get_size(image=get_value_at_index(constrainimagepysssss_795[0],0))
            get_image_size_end = time.time()
            print(f"Time taken to get image size: {get_image_size_end - get_image_size_start} seconds")

            load_intermediate_start = time.time()
            loadimage_812 = self.loadimage.load_image(image="intermediate_image.png")
            load_intermediate_end = time.time()
            print(f"Time taken to load intermediate image: {load_intermediate_end - load_intermediate_start} seconds")
            
            resize_intermediate_start = time.time()
            image_resize_814 = self.image_resize.image_rescale(
                mode="rescale",
                supersample="true",
                resampling="lanczos",
                rescale_factor=2.0000000000000004,
                resize_width=get_value_at_index(get_image_size_813, 0),
                resize_height=get_value_at_index(get_image_size_813, 1),
                image=get_value_at_index(loadimage_812, 0),
            )
            resize_intermediate_end = time.time()
            print(f"Time taken to resize intermediate image: {resize_intermediate_end - resize_intermediate_start} seconds")

            rmbg_start = time.time()
            rmbg_723 = self.rmbg.process_image(
                model="RMBG-2.0",
                sensitivity=1,
                process_res=1024,
                mask_blur=0,
                mask_offset=0,
                background="gray",
                invert_output=False,
                optimize="default",
                refine_foreground=False,
                image=get_value_at_index(constrainimagepysssss_795[0],0),
            )
            rmbg_end = time.time()
            print(f"Time taken for background removal: {rmbg_end - rmbg_start} seconds")

            segment_clothes_start = time.time()
            clothessegment_654 = self.clothessegment.segment_clothes(
                process_res=512,
                mask_blur=0,
                mask_offset=0,
                background_color="gray",
                invert_output=False,
                images=get_value_at_index(rmbg_723, 0),
                **class_selections
            )
            clothessegment_707 = self.clothessegment.segment_clothes(
                process_res=512,
                mask_blur=0,
                mask_offset=0,
                background_color="gray",
                invert_output=False,
                images=get_value_at_index(image_resize_814, 0),
                **class_selections
            )
            segment_clothes_end = time.time()
            print(f"Time taken for clothes segmentation: {segment_clothes_end - segment_clothes_start} seconds")

            mask_composite_start = time.time()
            maskcomposite_719 = self.maskcomposite.combine(
                x=0,
                y=0,
                operation="add",
                destination=get_value_at_index(clothessegment_654, 1),
                source=get_value_at_index(clothessegment_707, 1),
            )
            mask_composite_end = time.time()
            print(f"Time taken for mask composition: {mask_composite_end - mask_composite_start} seconds")

            mask_fix_start = time.time()
            maskfix_663 = self.maskfix.execute(
                erode_dilate=10,
                fill_holes=10,
                remove_isolated_pixels=10,
                smooth=10,
                blur=0,
                mask=get_value_at_index(maskcomposite_719, 0),
            )
            mask_fix_end = time.time()
            print(f"Time taken for mask fixing: {mask_fix_end - mask_fix_start} seconds")

            grow_mask_start = time.time()
            growmaskwithblur_798 = self.growmaskwithblur.expand_mask(
                expand=0,
                incremental_expandrate=1,
                tapered_corners=True,
                flip_input=False,
                blur_radius=6.800000000000001,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(maskfix_663, 0),
            )
            maskfix_810 = self.maskfix.execute(
                erode_dilate=0,
                fill_holes=0,
                remove_isolated_pixels=0,
                smooth=10,
                blur=0,
                mask=get_value_at_index(growmaskwithblur_798, 0),
            )
            grow_mask_end = time.time()
            print(f"Time taken for mask growing and final fixing: {grow_mask_end - grow_mask_start} seconds")
            
            intermediate_mask = get_value_at_index(maskfix_810, 0)
        mask_processing_end = time.time()
        print(f"Total time for mask processing: {mask_processing_end - mask_processing_start} seconds")

        rmbg_outfit_start = time.time()
        rmbg_710 = self.rmbg.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            background="gray",
            invert_output=False,
            optimize="default",
            refine_foreground=False,
            image=get_value_at_index(self.loadimage_23, 0),
        )
        rmbg_outfit_end = time.time()
        print(f"Time taken for outfit background removal: {rmbg_outfit_end - rmbg_outfit_start} seconds")

        gemini_prompt_start = time.time()
        # Replace LlamaVision with Gemini
        gemininode_main = self.gemininode.generate_content(
            prompt="You are a fashion expert and a prompt engineer, you describe the how clothes fit on a fashion model\n\n- strictly start the prompt with 'A fashion model wearing..' and describe the dress\n\n- describe about the clothes like shirt, pant or dress etc., precisely\n\n- include keywords to preserve cloth details, stitches etc.,\n\n- provide the prompt in paragraph style in less than 200 words",
            operation_mode="analysis",
            model_name="gemini-2.0-flash",
            temperature=0.3,
            seed=seed,
            sequential_generation=False,
            batch_count=1,
            aspect_ratio="none",
            external_api_key=GEMINI_API_KEY,
            chat_mode=False,
            clear_history=False,
            structured_output=False,
            max_images=6,
            max_output_tokens=256,
            use_random_seed=False,
            api_call_delay=1,
            images=get_value_at_index(rmbg_710, 0),
        )
        llm_prompt=get_value_at_index(gemininode_main, 0)
        gemini_prompt_end = time.time()
        print(f"Time taken to generate main LLM prompt: {gemini_prompt_end - gemini_prompt_start} seconds")

        print("model prompt: ", llm_prompt)
        
        df_text_start = time.time()
        df_text_281 = self.df_text.get_value(Text="dress, clothes")
        df_text_end = time.time()
        print(f"Time taken for df_text: {df_text_end - df_text_start} seconds")

        clip_encode_start = time.time()
        self.cliptextencode_120 = self.cliptextencode.encode(
            text=llm_prompt,
            clip=get_value_at_index(self.loraloader_606, 1),
        )

        self.cliptextencode_129 = self.cliptextencode.encode(
            text="", clip=get_value_at_index(self.loraloader_606, 1)
        )
        clip_encode_end = time.time()
        print(f"Time taken for CLIP text encoding: {clip_encode_end - clip_encode_start} seconds")

        flux_guidance_start = time.time()
        fluxguidance_130 = self.fluxguidance.append(
            guidance=30, conditioning=get_value_at_index(self.cliptextencode_120, 0)
        )
        flux_guidance_end = time.time()
        print(f"Time taken for flux guidance: {flux_guidance_end - flux_guidance_start} seconds")
        
        clip_vision_start = time.time()
        self.clipvisionencode_743 = self.clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(self.advancedvisionloader_741, 0),
            image=get_value_at_index(rmbg_710, 0),
        )
        clip_vision_end = time.time()
        print(f"Time taken for CLIP vision encoding: {clip_vision_end - clip_vision_start} seconds")

        style_model_start = time.time()
        stylemodelapply_742 = self.stylemodelapply.apply_stylemodel(
            strength=1.0000000000000002,
            strength_type="multiply",
            conditioning=get_value_at_index(fluxguidance_130, 0),
            style_model=get_value_at_index(self.stylemodelloader_163, 0),
            clip_vision_output=get_value_at_index(self.clipvisionencode_743, 0),
        )
        style_model_end = time.time()
        print(f"Time taken for style model application: {style_model_end - style_model_start} seconds")

        latent_prep_start = time.time()
        getimagesize_720 = self.getimagesize.execute(
            image=get_value_at_index(constrainimagepysssss_795[0],0)
        )

        emptylatentimage_349 = self.emptylatentimage.generate(
            width=get_value_at_index(getimagesize_720, 0),
            height=get_value_at_index(getimagesize_720, 1),
            batch_size=1,
        )

        df_get_latent_size_428 = self.df_get_latent_size.get_size(
            original=False, latent=get_value_at_index(emptylatentimage_349, 0)
        )

        imageresize_440 = self.imageresize.execute(
            width=get_value_at_index(df_get_latent_size_428, 0),
            height=get_value_at_index(df_get_latent_size_428, 1),
            interpolation="nearest",
            method="pad",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(rmbg_710, 0),
        )
        latent_prep_end = time.time()
        print(f"Time taken for latent preparation: {latent_prep_end - latent_prep_start} seconds")

        mask_resize_start = time.time()
        getimagesize_797 = self.getimagesize.execute(
            image=get_value_at_index(constrainimagepysssss_795[0],0)
        )

        resizemask_796 = self.resizemask.resize(
            width=get_value_at_index(getimagesize_797, 0),
            height=get_value_at_index(getimagesize_797, 1),
            keep_proportions=False,
            upscale_method="nearest-exact",
            crop="disabled",
            mask=intermediate_mask,
        )
        mask_resize_end = time.time()
        print(f"Time taken for mask resizing: {mask_resize_end - mask_resize_start} seconds")

        inpaint_crop_start = time.time()
        inpaintcropimproved = NODE_CLASS_MAPPINGS["InpaintCropImproved"]()
        inpaintcropimproved_799 = inpaintcropimproved.inpaint_crop(
            downscale_algorithm="lanczos",
            upscale_algorithm="lanczos",
            preresize=False,
            preresize_mode="ensure minimum resolution",
            preresize_min_width=1024,
            preresize_min_height=1024,
            preresize_max_width=16384,
            preresize_max_height=16384,
            mask_fill_holes=True,
            mask_expand_pixels=0,
            mask_invert=False,
            mask_blend_pixels=32,
            mask_hipass_filter=0.10000000000000002,
            extend_for_outpainting=False,
            extend_up_factor=1.0000000000000002,
            extend_down_factor=1.0000000000000002,
            extend_left_factor=1.0000000000000002,
            extend_right_factor=1.0000000000000002,
            context_from_mask_extend_factor=1.2000000000000002,
            output_resize_to_target_size=True,
            output_target_width=get_value_at_index(imageresize_440, 1),
            output_target_height=get_value_at_index(imageresize_440, 2),
            output_padding="32",
            image=get_value_at_index(constrainimagepysssss_795[0],0),
            mask=get_value_at_index(resizemask_796, 0),
        )
        inpaint_crop_end = time.time()
        print(f"Time taken for inpaint crop: {inpaint_crop_end - inpaint_crop_start} seconds")

        segment_anything_start = time.time()
        try:
            groundingdinosamsegment_segment_anything_441 = (
                self.groundingdinosam2segment_segment_anything2.main(
                    prompt=get_value_at_index(df_text_281, 0),
                    threshold=0.12000000000000002,
                    sam_model=get_value_at_index(
                        self.sam2modelloader_segment_anything2_8, 0
                    ),
                    grounding_dino_model=get_value_at_index(
                        self.groundingdinomodelloader_segment_anything2_9, 0
                    ),
                    image=get_value_at_index(imageresize_440, 0),
                )
            )
        except Exception as e:
            print("Error in groundingdinosam2segment_segment_anything2: ", e)
            groundingdinosamsegment_segment_anything_441 = (
                self.groundingdinosamsegment_segment_anything.main(
                    prompt=get_value_at_index(df_text_281, 0),
                threshold=0.3500000000000001,
                sam_model=get_value_at_index(self.sammodelloader_segment_anything_11, 0),
                grounding_dino_model=get_value_at_index(
                    self.groundingdinomodelloader_segment_anything_10, 0
                ),
                image=get_value_at_index(imageresize_440, 0),
            )
        )
        segment_anything_end = time.time()
        print(f"Time taken for segment anything: {segment_anything_end - segment_anything_start} seconds")

        image_processing_start = time.time()
        layerutility_imageremovealpha_430 = (
            self.layerutility_imageremovealpha.image_remove_alpha(
                fill_background=True,
                background_color="#ffffff",
                RGBA_image=get_value_at_index(
                    groundingdinosamsegment_segment_anything_441, 0
                ),
                mask=get_value_at_index(
                    groundingdinosamsegment_segment_anything_441, 1
                ),
            )
        )
        
        inpaintcrop_613 = self.inpaintcrop.inpaint_crop(
            context_expand_pixels=0,
            context_expand_factor=1.0000000000000002,
            fill_mask_holes=False,
            blur_mask_pixels=0,
            invert_mask=False,
            blend_pixels=16,
            rescale_algorithm="bislerp",
            mode="forced size",
            force_width=get_value_at_index(imageresize_440, 1),
            force_height=get_value_at_index(imageresize_440, 2),
            rescale_factor=1.0000000000000002,
            min_width=512,
            min_height=512,
            max_width=768,
            max_height=768,
            padding=32,
            image=get_value_at_index(layerutility_imageremovealpha_430, 0),
            mask=get_value_at_index(groundingdinosamsegment_segment_anything_441, 1),
        )

        imageconcanate_418 = self.imageconcanate.concatenate(
            direction="left",
            match_image_size=True,
            image1=get_value_at_index(inpaintcropimproved_799, 1),
            image2=get_value_at_index(inpaintcrop_613, 1),
        )

        masktoimage_424 = self.masktoimage.mask_to_image(
            mask=get_value_at_index(inpaintcropimproved_799, 2)
        )

        emptyimage_437 = self.emptyimage.generate(
            width=get_value_at_index(imageresize_440, 1),
            height=get_value_at_index(imageresize_440, 2),
            batch_size=1,
            color=0,
        )

        imageconcanate_419 = self.imageconcanate.concatenate(
            direction="left",
            match_image_size=True,
            image1=get_value_at_index(masktoimage_424, 0),
            image2=get_value_at_index(emptyimage_437, 0),
        )

        imagetomask_420 = self.imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imageconcanate_419, 0)
        )

        layermask_maskgrow_136 = self.layermask_maskgrow.mask_grow(
            invert_mask=False,
            grow=0,
            blur=0,
            mask=get_value_at_index(imagetomask_420, 0),
        )
        image_processing_end = time.time()
        print(f"Time taken for image processing: {image_processing_end - image_processing_start} seconds")

        inpaint_model_start = time.time()
        inpaintmodelconditioning_132 = self.inpaintmodelconditioning.encode(
            noise_mask=False,
            positive=get_value_at_index(stylemodelapply_742, 0),
            negative=get_value_at_index(self.cliptextencode_129, 0),
            vae=get_value_at_index(self.vaeloader_121, 0),
            pixels=get_value_at_index(imageconcanate_418, 0),
            mask=get_value_at_index(layermask_maskgrow_136, 0),
        )
        inpaint_model_end = time.time()
        print(f"Time taken for inpaint model conditioning: {inpaint_model_end - inpaint_model_start} seconds")
        
        text_encoder_conditioning = get_value_at_index(inpaintmodelconditioning_132, 0)

        if controlnet is not None:
            controlnet_start = time.time()
            print("-------Running Controlnet--------")
            preprocessor = "None"
            if controlnet == "depth":
                preprocessor = "DepthAnythingV2Preprocessor"
            elif controlnet == "openpose":
                preprocessor = "OpenposePreprocessor"
            elif controlnet == "canny":
                controlnet = "canny/lineart/anime_lineart/mlsd"
                preprocessor = "CannyEdgePreprocessor"
                
            setunioncontrolnettype_805 = self.setunioncontrolnettype.set_controlnet_type(
                type=controlnet, control_net=get_value_at_index(self.controlnetloader_804, 0)
            )

            aio_preprocessor_808 = self.aio_preprocessor.execute(
                preprocessor=preprocessor,
                resolution=512,
                image=get_value_at_index(imageconcanate_418, 0),
            )

            controlnetapplyadvanced_807 = self.controlnetapplyadvanced.apply_controlnet(
                strength=cn_strength,
                start_percent=cn_start,
                end_percent=cn_end,
                positive=get_value_at_index(inpaintmodelconditioning_132, 0),
                negative=get_value_at_index(inpaintmodelconditioning_132, 1),
                control_net=get_value_at_index(setunioncontrolnettype_805, 0),
                image=get_value_at_index(aio_preprocessor_808, 0),
                vae=get_value_at_index(self.vaeloader_121, 0),
            )
            controlnet_end = time.time()
            print(f"Time taken for controlnet processing: {controlnet_end - controlnet_start} seconds")
            text_encoder_conditioning = get_value_at_index(controlnetapplyadvanced_807, 0)

        noise_sampler_start = time.time()
        randomnoise_388 = self.randomnoise.get_noise(noise_seed=seed)
        ksamplerselect_390 = self.ksamplerselect.get_sampler(sampler_name="euler")
        simplemathint_603 = self.simplemathint.execute(value=30)
        noise_sampler_end = time.time()
        print(f"Time taken for noise and sampler setup: {noise_sampler_end - noise_sampler_start} seconds")

        guider_scheduler_start = time.time()
        basicguider_395 = self.basicguider.get_guider(
            model=get_value_at_index(self.differentialdiffusion_396, 0),
            conditioning=text_encoder_conditioning,
        )

        basicscheduler_391 = self.basicscheduler.get_sigmas(
            scheduler="simple",
            steps=get_value_at_index(simplemathint_603, 0),
            denoise=1,
            model=get_value_at_index(self.differentialdiffusion_396, 0),
        )
        guider_scheduler_end = time.time()
        print(f"Time taken for guider and scheduler setup: {guider_scheduler_end - guider_scheduler_start} seconds")

        sampling_start = time.time()
        samplercustomadvanced_387 = self.samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_388, 0),
            guider=get_value_at_index(basicguider_395, 0),
            sampler=get_value_at_index(ksamplerselect_390, 0),
            sigmas=get_value_at_index(basicscheduler_391, 0),
            latent_image=get_value_at_index(inpaintmodelconditioning_132, 2),
        )
        sampling_end = time.time()
        print(f"Time taken for sampling: {sampling_end - sampling_start} seconds")

        decode_start = time.time()
        vaedecode_134 = self.vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_387, 0),
            vae=get_value_at_index(self.vaeloader_121, 0),
        )
        decode_end = time.time()
        print(f"Time taken for VAE decoding: {decode_end - decode_start} seconds")

        crop_start = time.time()
        getimagesize_212 = self.getimagesize.execute(
            image=get_value_at_index(vaedecode_134, 0)
        )

        simplemath_213 = self.simplemath.execute(
            value="a/2", a=get_value_at_index(getimagesize_212, 0)
        )

        imagecrop_496 = self.imagecrop.execute(
            width=get_value_at_index(simplemath_213, 0),
            height=get_value_at_index(getimagesize_212, 1),
            position="top-left",
            x_offset=get_value_at_index(simplemath_213, 0),
            y_offset=0,
            image=get_value_at_index(vaedecode_134, 0),
        )
        crop_end = time.time()
        print(f"Time taken for image cropping: {crop_end - crop_start} seconds")

        stitch_start = time.time()
        inpaintstitchimproved_802 = self.inpaintstitchimproved.inpaint_stitch(
            stitcher=get_value_at_index(inpaintcropimproved_799, 0),
            inpainted_image=get_value_at_index(imagecrop_496, 0),
        )
        stitch_end = time.time()
        print(f"Time taken for inpaint stitching: {stitch_end - stitch_start} seconds")

        resize_start = time.time()
        imageresizekj_36 = self.imageresizekj.resize(
            width=512,
            height=512,
            upscale_method="nearest-exact",
            keep_proportion=False,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(inpaintstitchimproved_802, 0),
            get_image_size=get_value_at_index(self.loadimage_24, 0),
        )
        resize_end = time.time()
        print(f"Time taken for final image resizing: {resize_end - resize_start} seconds")
        
        final_image_start = time.time()
        if upscale:
            print("Upscaling final image...")
            final_image = self.imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(self.upscalemodelloader_14, 0),
                image=get_value_at_index(imageresizekj_36, 0),
            )
            print("Upscaling complete")
        else:
            final_image = imageresizekj_36

        for res in final_image[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        final_image_end = time.time()
        print(f"Time taken for final image processing: {final_image_end - final_image_start} seconds")

        full_end_time = time.time()
        print(f"Total time taken to generate final image: {full_end_time - full_start_time} seconds")
        return img


if __name__ == "__main__":
    model = SegfitCore()
    result = model(controlnet="depth",model_type="speed", seed=876456)
    result.save("result.png")
