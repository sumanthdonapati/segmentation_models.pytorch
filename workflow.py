import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np
import json
import time
import torch
import gc
from utilities import *

# similarity_model = ImageSimilarityModel()
# similarity_model.move_to_cpu()

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
    VAEEncode,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    CLIPTextEncode,
    VAEDecode,
    DualCLIPLoader,
    VAELoader,
    ControlNetLoader,
    ConditioningZeroOut,
    SetLatentNoiseMask,
    KSampler,
    LoraLoaderModelOnly,
    UNETLoader,
    ImageInvert,
    CheckpointLoaderSimple,
    ImageScaleBy,
    ControlNetApplyAdvanced,
    VAEEncodeTiled,
    VAEDecodeTiled,
    EmptyLatentImage
)


class DataWhisper:
    def __init__(self):
        import_custom_nodes()
        with torch.inference_mode():
            # Custom nodes
            self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
            self.inpaintresize = NODE_CLASS_MAPPINGS["InpaintResize"]()
            self.inpaintcrop = NODE_CLASS_MAPPINGS["InpaintCrop"]()
            self.text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
            self.load_lora = NODE_CLASS_MAPPINGS["Load Lora"]()
            self.setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
            self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            self.aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
            self.setshakkerlabsunioncontrolnettype = NODE_CLASS_MAPPINGS["SetShakkerLabsUnionControlNetType"]()
            self.cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
            self.easy_showanything = NODE_CLASS_MAPPINGS["easy showAnything"]()
            self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.stringfunctionpysssss = NODE_CLASS_MAPPINGS["StringFunction|pysssss"]()
            self.emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            self.controlnetinpaintingalimamaapply = NODE_CLASS_MAPPINGS["ControlNetInpaintingAliMamaApply"]()
            self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
            self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            self.constrainimagepysssss = NODE_CLASS_MAPPINGS["ConstrainImage|pysssss"]()
            self.controlnetapplysd3 = NODE_CLASS_MAPPINGS["ControlNetApplySD3"]()
            self.get_resolution_crystools = NODE_CLASS_MAPPINGS["Get resolution [Crystools]"]()
            self.yoloworld_modelloader_zho = NODE_CLASS_MAPPINGS["Yoloworld_ModelLoader_Zho"]()
            self.bboxdetectorcombined_v2 = NODE_CLASS_MAPPINGS["BboxDetectorCombined_v2"]()
            self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            self.segmdetectorcombined_v2 = NODE_CLASS_MAPPINGS["SegmDetectorCombined_v2"]()
            self.maskfix = NODE_CLASS_MAPPINGS["MaskFix+"]()
            self.subtractmask = NODE_CLASS_MAPPINGS["SubtractMask"]()
            self.segmdetectorsegs = NODE_CLASS_MAPPINGS["SegmDetectorSEGS"]()
            self.segstocombinedmask = NODE_CLASS_MAPPINGS["SegsToCombinedMask"]()
            self.separate_mask_components = NODE_CLASS_MAPPINGS["Separate Mask Components"]()
            self.esam_modelloader_zho = NODE_CLASS_MAPPINGS["ESAM_ModelLoader_Zho"]()
            self.yoloworld_esam_detectorprovider_zho = NODE_CLASS_MAPPINGS["Yoloworld_ESAM_DetectorProvider_Zho"]()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.inpaintstitch = NODE_CLASS_MAPPINGS["InpaintStitch"]()
            self.cr_apply_lora_stack = NODE_CLASS_MAPPINGS["CR Apply LoRA Stack"]()
            self.cr_lora_stack = NODE_CLASS_MAPPINGS["CR LoRA Stack"]()
            self.ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
            self.tobasicpipe = NODE_CLASS_MAPPINGS["ToBasicPipe"]()
            self.frombasicpipe = NODE_CLASS_MAPPINGS["FromBasicPipe"]()
            self.tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
            self.freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()

            # Standard nodes
            self.conditioningzeroout = ConditioningZeroOut()
            self.cliptextencode = CLIPTextEncode()
            self.controlnetloader = ControlNetLoader()
            self.imageinvert = ImageInvert()
            self.vaeloader = VAELoader()
            self.vaeencode = VAEEncode()
            self.unetloader = UNETLoader()
            self.loraloadermodelonly = LoraLoaderModelOnly()
            self.dualcliploader = DualCLIPLoader()
            self.setlatentnoisemask = SetLatentNoiseMask()
            self.checkpointloadersimple = CheckpointLoaderSimple()
            self.controlnetapplyadvanced = ControlNetApplyAdvanced()
            self.emptylatentimage = EmptyLatentImage()
            self.ksampler = KSampler()
            self.vaedecode = VAEDecode()
            self.loadimage = LoadImage()
            self.imagescaleby = ImageScaleBy()
            self.vaeencodetiled = VAEEncodeTiled()
            self.vaedecodetiled = VAEDecodeTiled()

            # Load initial models and configurations
            self.checkpointloadersimple_4 = self.checkpointloadersimple.load_checkpoint(ckpt_name="juggernaut_reborn.safetensors")
            self.cr_lora_stack_13 = self.cr_lora_stack.lora_stacker(
                switch_1="On",
                lora_name_1="add-details.safetensors",
                model_weight_1=1,
                clip_weight_1=1,
                switch_2="On",
                lora_name_2="SDXLrender_v2.0.safetensors",
                model_weight_2=1,
                clip_weight_2=1,
                switch_3="Off",
                lora_name_3="None",
                model_weight_3=1,
                clip_weight_3=1,
            )
            self.cr_apply_lora_stack_14 = self.cr_apply_lora_stack.apply_lora_stack(
                model=get_value_at_index(self.checkpointloadersimple_4, 0),
                clip=get_value_at_index(self.checkpointloadersimple_4, 1),
                lora_stack=get_value_at_index(self.cr_lora_stack_13, 0),
            )
            self.controlnetloader_20 = self.controlnetloader.load_controlnet(
                control_net_name="control_v11f1e_sd15_tile_fp16.safetensors"
            )
            self.upscalemodelloader_53 = self.upscalemodelloader.load_model(
                model_name="4x-UltraSharp.pth"
            )
            self.unetloader_24 = self.unetloader.load_unet(
                unet_name="FLUX1/flux1-dev-fp8.safetensors", 
                weight_dtype="fp8_e4m3fn"
            )
            self.loraloadermodelonly_4 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="flux-dev-alpha-turbo.safetensors",
                strength_model=1,
                model=get_value_at_index(self.unetloader_24, 0),
            )
            self.upscalemodelloader_12 = self.upscalemodelloader.load_model(
                model_name="4x-UltraSharp.pth"
            )
            self.dualcliploader_23 = self.dualcliploader.load_clip(
                clip_name1="t5xxl_fp8_e4m3fn.safetensors",
                clip_name2="clip_l.safetensors",
                type="flux",
            )
            self.vaeloader_22 = self.vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")
            self.controlnetloader_30 = self.controlnetloader.load_controlnet(
                control_net_name="flux-dev-controlnet-inpainting-beta.safetensors"
            )
            controlnetloader_13 = self.controlnetloader.load_controlnet(
                control_net_name="FLUX.1/Shakker-Labs-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors"
            )
            self.setunioncontrolnettype_25 = self.setunioncontrolnettype.set_controlnet_type(
                type="canny/lineart/anime_lineart/mlsd",
                control_net=get_value_at_index(controlnetloader_13, 0),
            )
            self.yoloworld_modelloader_zho_48 = self.yoloworld_modelloader_zho.load_yolo_world_model(
                yolo_world_model="yolo_world/l"
            )
            self.esam_modelloader_zho_49 = self.esam_modelloader_zho.load_esam_model(device="CUDA")
            self.ultralyticsdetectorprovider_72 = self.ultralyticsdetectorprovider.doit(
                model_name="segm/yolo11x-seg.pt"
            )
            self.checkpointloadersimple_283 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name="juggxlv11_lightning.safetensors"
            )
            self.controlnetloader_48 = self.controlnetloader.load_controlnet(
                control_net_name="SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
            )
            self.setunioncontrolnettype_294 = self.setunioncontrolnettype.set_controlnet_type(
                type="depth", 
                control_net=get_value_at_index(self.controlnetloader_48, 0)
            )
            self.setshakkerlabsunioncontrolnettype_258 = self.setshakkerlabsunioncontrolnettype.set_controlnet_type(
                type="canny", 
                control_net=get_value_at_index(self.controlnetloader_48, 0)
            )

            # Load JSON configurations
            with open("lora_prompts.json", 'r') as file:
                self.lora_mapped_prompts = json.load(file)
            with open("segment_maps.json", 'r') as file:
                self.segment_maps = json.load(file)
         
    def crop_inpaint_func(self, lora_name, lora_prompt):
        
        lora_name = lora_name
        lora_prompt = lora_prompt
        loadimage_37 = self.loadimage.load_image(
                image="inpaint_image.png"
            ) #image
        loadimage_44 = self.loadimage.load_image(image="inpaint_mask.png")
        imagetomask_54 = self.imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(loadimage_44, 0)
        )
        image_tensor = get_value_at_index(loadimage_37, 0)
        mask_tensor = get_value_at_index(imagetomask_54, 0)
        print('image tensor size:',image_tensor.shape)
        print('mask tensor size:',mask_tensor.shape)
        try:
            inpaintresize_4 = self.inpaintresize.inpaint_resize(
                rescale_algorithm="bicubic",
                mode="ensure minimum size",
                min_width=1024,
                min_height=1024,
                rescale_factor=1,
                image=get_value_at_index(loadimage_37, 0),
                mask=get_value_at_index(imagetomask_54, 0),
            )
            inpaintcrp_image = get_value_at_index(inpaintresize_4, 0)
            inpaintcrp_mask = get_value_at_index(inpaintresize_4, 1)
        except Exception as e:
            print(f"Error in inpaintresize: {e}")
            inpaintcrp_image = get_value_at_index(loadimage_37, 0)
            inpaintcrp_mask = get_value_at_index(imagetomask_54, 0)
        inpaintcrop_3 = self.inpaintcrop.inpaint_crop(
            context_expand_pixels=351,
            context_expand_factor=1,
            fill_mask_holes=True,
            blur_mask_pixels=17.900000000000002,
            invert_mask=False,
            blend_pixels=16,
            rescale_algorithm="bicubic",
            mode="ranged size",
            force_width=1024,
            force_height=1024,
            rescale_factor=1,
            min_width=512,
            min_height=512,
            max_width=768,
            max_height=768,
            padding=32,
            image=inpaintcrp_image,
            mask=inpaintcrp_mask,
        )
        
        vaeencode_9 = self.vaeencode.encode(
            pixels=get_value_at_index(inpaintcrop_3, 1),
            vae=get_value_at_index(self.vaeloader_22, 0),
        )
        
        #lora_name
        load_lora_33 = self.load_lora.load_lora(
            lora_name=f"Furniture lora/{lora_name}.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(self.loraloadermodelonly_4, 0),
            clip=get_value_at_index(self.dualcliploader_23, 0),
        )

        #prompt
        text_multiline_34 = self.text_multiline.text_multiline(
            text=lora_prompt
        )

        stringfunctionpysssss_36 = self.stringfunctionpysssss.exec(
            action="append",
            tidy_tags="yes",
            text_a=get_value_at_index(load_lora_33, 2),
            text_b=get_value_at_index(text_multiline_34, 0),
        )

        cliptextencode_17 = self.cliptextencode.encode(
            text=get_value_at_index(stringfunctionpysssss_36, 0),
            clip=get_value_at_index(load_lora_33, 1),
        )

        conditioningzeroout_18 = self.conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_17, 0)
        )

        growmaskwithblur_10 = self.growmaskwithblur.expand_mask(
            expand=2,
            incremental_expandrate=0,
            tapered_corners=False,
            flip_input=False,
            blur_radius=2,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
            mask=get_value_at_index(inpaintcrop_3, 2),
        )

        controlnetinpaintingalimamaapply_2 = (
            self.controlnetinpaintingalimamaapply.apply_inpaint_controlnet(
                strength=1,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(cliptextencode_17, 0),
                negative=get_value_at_index(conditioningzeroout_18, 0),
                control_net=get_value_at_index(self.controlnetloader_30, 0),
                vae=get_value_at_index(self.vaeloader_22, 0),
                image=get_value_at_index(inpaintcrop_3, 1),
                mask=get_value_at_index(growmaskwithblur_10, 0),
            )
        )

        fluxguidance_12 = self.fluxguidance.append(
            guidance=3.5,
            conditioning=get_value_at_index(controlnetinpaintingalimamaapply_2, 0),
        )

        setlatentnoisemask_8 = self.setlatentnoisemask.set_mask(
            samples=get_value_at_index(vaeencode_9, 0),
            mask=get_value_at_index(growmaskwithblur_10, 0),
        )

        ksampler_1 = self.ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(load_lora_33, 0),
            positive=get_value_at_index(fluxguidance_12, 0),
            negative=get_value_at_index(controlnetinpaintingalimamaapply_2, 1),
            latent_image=get_value_at_index(setlatentnoisemask_8, 0),
        )

        vaedecode_5 = self.vaedecode.decode(
            samples=get_value_at_index(ksampler_1, 0),
            vae=get_value_at_index(self.vaeloader_22, 0),
        )

        inpaintstitch_7 = self.inpaintstitch.inpaint_stitch(
            rescale_algorithm="bislerp",
            stitch=get_value_at_index(inpaintcrop_3, 0),
            inpainted_image=get_value_at_index(vaedecode_5, 0),
        )
        
        for res in inpaintstitch_7[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        img.save("ComfyUI/input/inpaint_image.png")
        return img
    
    @torch.inference_mode()
    def upscale_image(self):
        # Load and encode initial image and text
        loadimage_17 = self.loadimage.load_image(image="inpaint_image.png")
        
        cliptextencode_83 = self.cliptextencode.encode(
            text="masterpiece, best quality, highres",
            clip=get_value_at_index(self.cr_apply_lora_stack_14, 1),
        )
        
        cliptextencode_7 = self.cliptextencode.encode(
            text="(worst quality, low quality, normal quality:2) embedding:JuggernautNegative-neg, ",
            clip=get_value_at_index(self.cr_apply_lora_stack_14, 1),
        )

        # Process the initial image
        constrainimagepysssss_86 = self.constrainimagepysssss.constrain_image(
            max_width=1024,
            max_height=1024,
            min_width=0,
            min_height=0,
            crop_if_required="no",
            images=get_value_at_index(loadimage_17, 0),
        )

        # Set up the basic pipeline
        tobasicpipe_24 = self.tobasicpipe.doit(
            model=get_value_at_index(self.cr_apply_lora_stack_14, 0),
            clip=get_value_at_index(self.cr_apply_lora_stack_14, 1),
            vae=get_value_at_index(self.checkpointloadersimple_4, 2),
            positive=get_value_at_index(cliptextencode_83, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
        )
        
        frombasicpipe_21 = self.frombasicpipe.doit(
            basic_pipe=get_value_at_index(tobasicpipe_24, 0)
        )

        # Apply ControlNet
        controlnetapplyadvanced_19 = self.controlnetapplyadvanced.apply_controlnet(
            strength=0.9,
            start_percent=0,
            end_percent=0.9,
            positive=get_value_at_index(frombasicpipe_21, 3),
            negative=get_value_at_index(frombasicpipe_21, 4),
            control_net=get_value_at_index(self.controlnetloader_20, 0),
            image=get_value_at_index(constrainimagepysssss_86[0], 0),
        )

        # Update the basic pipeline
        tobasicpipe_23 = self.tobasicpipe.doit(
            model=get_value_at_index(frombasicpipe_21, 0),
            clip=get_value_at_index(frombasicpipe_21, 1),
            vae=get_value_at_index(frombasicpipe_21, 2),
            positive=get_value_at_index(controlnetapplyadvanced_19, 0),
            negative=get_value_at_index(frombasicpipe_21, 4),
        )

        frombasicpipe_36 = self.frombasicpipe.doit(
            basic_pipe=get_value_at_index(tobasicpipe_23, 0)
        )

        # Upscale and process the image
        imageupscalewithmodel_52 = self.imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(self.upscalemodelloader_53, 0),
            image=get_value_at_index(loadimage_17, 0),
        )

        imagescaleby_54 = self.imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=0.17,
            image=get_value_at_index(imageupscalewithmodel_52, 0),
        )

        # Encode the image
        vaeencodetiled_46 = self.vaeencodetiled.encode(
            tile_size=768,
            pixels=get_value_at_index(imagescaleby_54, 0),
            vae=get_value_at_index(frombasicpipe_36, 2),
        )
        
        # Apply tiled diffusion and FreeU
        tileddiffusion_44 = self.tileddiffusion.apply(
            method="MultiDiffusion",
            tile_width=768,
            tile_height=768,
            tile_overlap=64,
            tile_batch_size=8,
            model=get_value_at_index(frombasicpipe_36, 0),
        )

        freeu_v2_55 = self.freeu_v2.patch(
            b1=1.3,
            b2=1.4000000000000001,
            s1=0.9,
            s2=0.2,
            model=get_value_at_index(tileddiffusion_44, 0),
        )

        # Sample and decode
        ksampler_45 = self.ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=20,
            cfg=5,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            denoise=0.3,
            model=get_value_at_index(freeu_v2_55, 0),
            positive=get_value_at_index(frombasicpipe_36, 3),
            negative=get_value_at_index(frombasicpipe_36, 4),
            latent_image=get_value_at_index(vaeencodetiled_46, 0),
        )

        vaedecodetiled_48 = self.vaedecodetiled.decode(
            tile_size=768,
            samples=get_value_at_index(ksampler_45, 0),
            vae=get_value_at_index(frombasicpipe_36, 2),
        )
        
        for res in vaedecodetiled_48[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        return [img]
    
    @torch.inference_mode()
    def sketch2image(self, prompt):
        loadimage_295 = self.loadimage.load_image(image="sketch_image.png")
        cliptextencode_6 = self.cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(self.checkpointloadersimple_283, 1),
        )

        cliptextencode_7 = self.cliptextencode.encode(
            text="CGI, Unreal, Airbrushed, Digital, Blurry image",
            clip=get_value_at_index(self.checkpointloadersimple_283, 1),
        )

        imageresizekj_301 = self.imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(loadimage_295, 0),
        )

        aio_preprocessor_31 = self.aio_preprocessor.execute(
            preprocessor="Zoe-DepthMapPreprocessor",
            resolution=1024,
            image=get_value_at_index(imageresizekj_301, 0),
        )

        controlnetapplyadvanced_47 = self.controlnetapplyadvanced.apply_controlnet(
            strength=0.5,
            start_percent=0,
            end_percent=0.8,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            control_net=get_value_at_index( self.setunioncontrolnettype_294, 0),
            image=get_value_at_index(aio_preprocessor_31, 0),
            vae=get_value_at_index( self.checkpointloadersimple_283, 2),
        )



        cannyedgepreprocessor_271 =  self.cannyedgepreprocessor.execute(
            low_threshold=80,
            high_threshold=230,
            resolution=1024,
            image=get_value_at_index(imageresizekj_301, 0),
        )

        controlnetapplyadvanced_269 =  self.controlnetapplyadvanced.apply_controlnet(
            strength=1,
            start_percent=0,
            end_percent=0.9,
            positive=get_value_at_index(controlnetapplyadvanced_47, 0),
            negative=get_value_at_index(controlnetapplyadvanced_47, 1),
            control_net=get_value_at_index(
                    self.setshakkerlabsunioncontrolnettype_258, 0
            ),
            image=get_value_at_index(cannyedgepreprocessor_271, 0),
            vae=get_value_at_index( self.checkpointloadersimple_283, 2),
        )

        emptylatentimage_298 =  self.emptylatentimage.generate(
            width=get_value_at_index(imageresizekj_301, 1),
            height=get_value_at_index(imageresizekj_301, 2),
            batch_size=1,
        )

        ksampler_3 =  self.ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=2,
            sampler_name="dpmpp_sde_gpu",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index( self.checkpointloadersimple_283, 0),
            positive=get_value_at_index(controlnetapplyadvanced_269, 0),
            negative=get_value_at_index(controlnetapplyadvanced_269, 1),
            latent_image=get_value_at_index(emptylatentimage_298, 0),
        )

        vaedecode_8 =  self.vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index( self.checkpointloadersimple_283, 2),
        )
        for res in vaedecode_8[0]:
            img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            # img.save('sketch2final2.png')
        
        return img
       
    @torch.inference_mode()
    def create_mask(self, lora_name):
        
        segment_word = self.segment_maps[lora_name]
        all_words = 'bed, chair, lamp, side table, lamp, books, flowers, vase'
        other_words = all_words.replace(segment_word,'')
        print("segment word: ",segment_word)
        print("other word: ",other_words)
        loadimage_51 = self.loadimage.load_image(
            image="inpaint_image.png"
        )
        
        yoloworld_esam_detectorprovider_zho_47 = (
            self.yoloworld_esam_detectorprovider_zho.doit(
                categories=segment_word,
                iou_threshold=0.1,
                with_class_agnostic_nms=False,
                yolo_world_model=get_value_at_index(
                    self.yoloworld_modelloader_zho_48, 0
                ),
                esam_model_opt=get_value_at_index(self.esam_modelloader_zho_49, 0),
            )
        )

        bboxdetectorcombined_v2_50 = self.bboxdetectorcombined_v2.doit(
            threshold=0.5,
            dilation=4,
            bbox_detector=get_value_at_index(
                yoloworld_esam_detectorprovider_zho_47, 0
            ),
            image=get_value_at_index(loadimage_51, 0),
        )

        yoloworld_esam_detectorprovider_zho_52 = (
            self.yoloworld_esam_detectorprovider_zho.doit(
                categories=other_words,
                iou_threshold=0.1,
                with_class_agnostic_nms=False,
                yolo_world_model=get_value_at_index(
                    self.yoloworld_modelloader_zho_48, 0
                ),
                esam_model_opt=get_value_at_index(self.esam_modelloader_zho_49, 0),
            )
        )

        segmdetectorcombined_v2_53 = self.segmdetectorcombined_v2.doit(
            threshold=0.7,
            dilation=0,
            segm_detector=get_value_at_index(
                yoloworld_esam_detectorprovider_zho_52, 1
            ),
            image=get_value_at_index(loadimage_51, 0),
        )

        #yoloworld segment detector
        if segment_word != "bed":
            # if segment_word == "chair":
            #     erode_value = 5
            # else:
            #     erode_value = 5
            segmdetectorcombined_v2_54 = self.segmdetectorcombined_v2.doit(
                threshold=0.5,
                dilation=0,
                segm_detector=get_value_at_index(
                    yoloworld_esam_detectorprovider_zho_47, 1
                ),
                image=get_value_at_index(loadimage_51, 0),
            )

            maskfix_56 = self.maskfix.execute(
                erode_dilate=3,
                fill_holes=20,
                remove_isolated_pixels=20,
                smooth=10,
                blur=0,
                mask=get_value_at_index(segmdetectorcombined_v2_54, 0),
            )
        #yolov11 detector for bed
        else:
            segmdetectorsegs_73 = self.segmdetectorsegs.doit(
            threshold=0.5,
            dilation=10,
            crop_factor=3,
            drop_size=10,
            labels=segment_word,
            segm_detector=get_value_at_index(self.ultralyticsdetectorprovider_72, 1),
            image=get_value_at_index(loadimage_51, 0),
            )

            segstocombinedmask_74 = self.segstocombinedmask.doit(
                segs=get_value_at_index(segmdetectorsegs_73, 0)
            )
            maskfix_56 = self.maskfix.execute(
                erode_dilate=5,
                fill_holes=20,
                remove_isolated_pixels=10,
                smooth=10,
                blur=0,
                mask=get_value_at_index(segstocombinedmask_74, 0),
            )

        # subtractmask_55 = self.subtractmask.doit(
        #     mask1=get_value_at_index(maskfix_56, 0),
        #     mask2=get_value_at_index(segmdetectorcombined_v2_53, 0),
        # )

        # maskfix_57 = self.maskfix.execute(
        #     erode_dilate=0,
        #     fill_holes=30,
        #     remove_isolated_pixels=30,
        #     smooth=50,
        #     blur=0,
        #     mask=get_value_at_index(subtractmask_55, 0),
        # )

        # subtractmask_58 = self.subtractmask.doit(
        #     mask1=get_value_at_index(bboxdetectorcombined_v2_50, 0),
        #     mask2=get_value_at_index(maskfix_57, 0),
        # )

        # subtractmask_59 = self.subtractmask.doit(
        #     mask1=get_value_at_index(bboxdetectorcombined_v2_50, 0),
        #     mask2=get_value_at_index(subtractmask_58, 0),
        # )
        # separate_mask_components_43 = self.separate_mask_components.separate(
        #         mask=get_value_at_index(subtractmask_59, 0)
            # )
        # if segment_word == "bed":
        #     if len(separate_mask_components_43[0]) == 0:
        #         final_image = subtractmask_59
        #     else:
        #         final_image = separate_mask_components_43  
        # else:
        separate_mask_components_44 = self.separate_mask_components.separate(
            mask=get_value_at_index(maskfix_56, 0)
        )
        if len(separate_mask_components_44[0]) == 0:
            final_image = maskfix_56
        else:
            final_image = separate_mask_components_44
                
        return final_image
        
    @torch.inference_mode()
    def __call__(self, *args,  **kwargs):
        pre_start = time.time()
        prompt = kwargs.get('prompt', "A cozy, modern bedroom...")
        prompt = prompt + ", high quality, 4k, HD"
        upscale = kwargs.get('upscale',False)
        
        # Get all loras mentioned in the prompt
        all_loras = list(self.lora_mapped_prompts.keys())
        mentioned_loras = [lora for lora in all_loras if lora.lower() in prompt.lower()]
        
        # Define the desired order of furniture types based on segment_maps.json
        furniture_order = ['bed', 'chair', 'side table', 'table', 'bench','lamp']
        
        # Automatically create furniture groups based on segment_maps, maintaining order
        furniture_groups = {}
        # First initialize all possible furniture types in order
        for furniture_type in furniture_order:
            furniture_groups[furniture_type] = []
            
        # Then populate with mentioned loras
        for lora in mentioned_loras:
            furniture_type = self.segment_maps[lora]
            furniture_groups[furniture_type].append(lora)
            
        # Remove empty furniture groups
        furniture_groups = {k: v for k, v in furniture_groups.items() if v}
        
        print("Furniture groups detected:", {k: len(v) for k, v in furniture_groups.items()})
        # Generate initial sketch
        print("preprocess time: ",time.time()-pre_start)
        print("generating sketch image...")
        real_img = self.sketch2image(prompt)
        real_img.save("ComfyUI/input/inpaint_image.png")
        real_img.save("ComfyUI/input/s2i_image.png")
        print("sketch2image time: ", time.time()-pre_start)
        
        # Process each furniture group
        for furniture_type, loras in furniture_groups.items():
            mask_start = time.time()
            print(f"\nProcessing {furniture_type} group with {len(loras)} loras")
            
            # If multiple furniture of same type, do similarity matching
            if len(loras) > 1:
                mask_start=time.time()
                masks = self.create_mask(loras[0])  # Use any lora of this type to create masks
                mask_end=time.time()
                print("mask process time:",mask_end-mask_start)
                # Load reference images for each lora
                reference_images = {}
                for lora in loras:
                    try:
                        ref_img = load_and_preprocess(f"reference_images/{lora}.png")
                        reference_images[lora] = ref_img
                    except Exception as e:
                        print(f"Warning: Could not load reference image for {lora}: {e}")
                
                # Process each mask
                for i, mask in enumerate(masks[0]):
                    mask_array = mask.detach().cpu().numpy().squeeze()
                    if np.all(mask_array == 0):
                        continue
                    
                    # Convert mask to image for comparison
                    mask_img = Image.fromarray(np.clip(255. * mask_array, 0, 255).astype(np.uint8))
                    # mask_img.save(f"test_imgs/mask_{furniture_type}_{i}.png")
                    
                    # Find best matching lora for this mask
                    best_match = None
                    best_score = 0
                    composite_image = crop_composite_and_resize(real_img,mask_img)
                    composite_image.save(f'composite_image.png')
                    for lora, ref_img in reference_images.items():
                        # similarity_model.move_to_gpu()
                        similarity = compare_products(composite_image, ref_img)
                        # similarity_model.move_to_cpu()
                        print(f"{lora} similarity:",similarity)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = lora
                    
                    print(f"Mask {i} best matches with {best_match} (score: {best_score:.2f})")
                    
                    # Apply the matched lora
                    if best_match:
                        mask_img.save("ComfyUI/input/inpaint_mask.png")
                        print('mask img shape:',mask_img.size)
                        print('real img shape:',real_img.size)
                        inpaint_start = time.time()
                        self.crop_inpaint_func(best_match, self.lora_mapped_prompts[best_match])
                        inpaint_end = time.time()
                        print("inpaint time:",inpaint_end-inpaint_start)
            # If single furniture, process normally
            else:
                lora = loras[0]
                mask_start=time.time()
                masks = self.create_mask(lora)
                mask_end=time.time()
                print("mask process time:",mask_end-mask_start)
                for i, mask in enumerate(masks[0]):
                    mask_array = mask.detach().cpu().numpy().squeeze()
                    if np.all(mask_array == 0):
                        continue
                    mask_img = Image.fromarray(np.clip(255. * mask_array, 0, 255).astype(np.uint8))
                    mask_img.save("ComfyUI/input/inpaint_mask.png")
                    # mask_img.save(f'test_imgs/mask_{lora}_{i}.png')
                    inpaint_start = time.time()
                    self.crop_inpaint_func(lora, self.lora_mapped_prompts[lora])
                    inpaint_end = time.time()
                    print("inpaint time:",inpaint_end-inpaint_start)
            
            print(f"{furniture_type} processing time: ", time.time() - mask_start)
            torch.cuda.empty_cache()
        
        if upscale:
            final_img = self.upscale_image()
        else:
            final_img = Image.open('ComfyUI/input/inpaint_image.png')
        
        torch.cuda.empty_cache()
        gc.collect()
        return [final_img]
