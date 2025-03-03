import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np

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
    CLIPSetLastLayer,
    CheckpointLoaderSimple,
    VAEDecode,
    SetLatentNoiseMask,
    KSampler,
    LoadImage,
    VAEEncode,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
)


class ObjectRemovalWorkflow:
    def __init__(self):
        import_custom_nodes()
        # Get all required node mappings
        self.groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoModelLoader (segment anything)"]()
        self.sammodelloader_segment_anything = NODE_CLASS_MAPPINGS["SAMModelLoader (segment anything)"]()
        self.getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        self.groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS["GroundingDinoSAMSegment (segment anything)"]()
        self.impactdilatemask = NODE_CLASS_MAPPINGS["ImpactDilateMask"]()
        self.growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        self.lamaremover = NODE_CLASS_MAPPINGS["LamaRemover"]()
        self.primitive_float_crystools = NODE_CLASS_MAPPINGS["Primitive float [Crystools]"]()
        self.ttn_seed = NODE_CLASS_MAPPINGS["ttN seed"]()
        self.differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        self.impactgaussianblurmask = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
        self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        
        # Initialize built-in nodes
        self.checkpointloadersimple = CheckpointLoaderSimple()
        self.clipsetlastlayer = CLIPSetLastLayer()
        self.cliptextencode = CLIPTextEncode()
        self.loadimage = LoadImage()
        self.vaeencode = VAEEncode()
        self.setlatentnoisemask = SetLatentNoiseMask()
        self.ksampler = KSampler()
        self.vaedecode = VAEDecode()

        # Load models that only need to be loaded once
        with torch.inference_mode():
            self.checkpointloadersimple_1 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name="RealVisXL_V5_Lightning.safetensors"
            )

            self.groundingdinomodelloader_segment_anything_22 = (
                self.groundingdinomodelloader_segment_anything.main(
                    model_name="GroundingDINO_SwinB (938MB)"
                )
            )

            self.sammodelloader_segment_anything_23 = self.sammodelloader_segment_anything.main(
                model_name="sam_vit_h (2.56GB)"
            )

    def __call__(self, *args, **kwargs):
        with torch.inference_mode():
            sam_prompt = kwargs.get("sam_prompt")
            sam_threshold = kwargs.get("sam_threshold", 0.3)
            text_prompt = kwargs.get("text_prompt", "remove,empty background, keep the context of the surrounding")
            seed = kwargs.get("seed", random.randint(1, 2**32))
            loadimage_87 = self.loadimage.load_image(
                image="input_image.png"
            )
            getimagesizeandcount_179 = self.getimagesizeandcount.getsize(
                image=get_value_at_index(loadimage_87, 0)
            )

            imageresizekj_176 = self.imageresizekj.resize(
                width=1280,
                height=1280,
                upscale_method="lanczos",
                keep_proportion=True,
                divisible_by=2,
                crop="disabled",
                image=get_value_at_index(getimagesizeandcount_179, 0),
            )
            print("sam_prompt: ",sam_prompt)
            if sam_prompt is not None:
                groundingdinosamsegment_segment_anything_20 = (
                    self.groundingdinosamsegment_segment_anything.main(
                        prompt=sam_prompt,
                        threshold=sam_threshold,
                        sam_model=get_value_at_index(self.sammodelloader_segment_anything_23, 0),
                        grounding_dino_model=get_value_at_index(
                            self.groundingdinomodelloader_segment_anything_22, 0
                        ),
                        image=get_value_at_index(imageresizekj_176, 0),
                    )
                )
                impactdilatemask_26 = self.impactdilatemask.doit(
                dilation=15,
                mask=get_value_at_index(groundingdinosamsegment_segment_anything_20, 1),
                )
            else:
                loadimage_174 = self.loadimage.load_image(
                    image="input_mask.jpg"
                )

                imageresizekj_177 = self.imageresizekj.resize(
                    width=get_value_at_index(imageresizekj_176, 1),
                    height=get_value_at_index(imageresizekj_176, 2),
                    upscale_method="lanczos",
                    keep_proportion=True,
                    divisible_by=2,
                    crop="disabled",
                    image=get_value_at_index(loadimage_174, 0),
                )

                imagetomask_175 = self.imagetomask.image_to_mask(
                    channel="red", image=get_value_at_index(imageresizekj_177, 0)
                )
                impactdilatemask_26 = self.impactdilatemask.doit(
                dilation=15,
                mask=get_value_at_index(imagetomask_175, 0),
                )

            clipsetlastlayer_2 = self.clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-1, clip=get_value_at_index(self.checkpointloadersimple_1, 1)
            )

            cliptextencode_4 = self.cliptextencode.encode(
                text=text_prompt,
                clip=get_value_at_index(clipsetlastlayer_2, 0),
            )

            growmask_54 = self.growmask.expand_mask(
                expand=20,
                tapered_corners=True,
                mask=get_value_at_index(impactdilatemask_26, 0),
            )

            lamaremover_27 = self.lamaremover.lama_remover(
                mask_threshold=110,
                gaussblur_radius=10,
                invert_mask=False,
                images=get_value_at_index(imageresizekj_176, 0),
                masks=get_value_at_index(growmask_54, 0),
            )

            vaeencode_30 = self.vaeencode.encode(
                pixels=get_value_at_index(lamaremover_27, 0),
                vae=get_value_at_index(self.checkpointloadersimple_1, 2),
            )

            cliptextencode_32 = self.cliptextencode.encode(
                text=text_prompt,
                clip=get_value_at_index(self.checkpointloadersimple_1, 1),
            )

            primitive_float_crystools_97 = self.primitive_float_crystools.execute(float=0.6)

            ttn_seed_163 = self.ttn_seed.plant(seed=seed)

            setlatentnoisemask_29 = self.setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencode_30, 0),
                mask=get_value_at_index(impactdilatemask_26, 0),
            )

            differentialdiffusion_171 = self.differentialdiffusion.apply(
                model=get_value_at_index(self.checkpointloadersimple_1, 0)
            )

            ksampler_31 = self.ksampler.sample(
                seed=seed,
                steps=6,
                cfg=1.2,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                denoise=0.35000000000000003,
                model=get_value_at_index(differentialdiffusion_171, 0),
                positive=get_value_at_index(cliptextencode_32, 0),
                negative=get_value_at_index(cliptextencode_4, 0),
                latent_image=get_value_at_index(setlatentnoisemask_29, 0),
            )

            ksampler_36 = self.ksampler.sample(
                seed=seed,
                steps=4,
                cfg=1,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                denoise=0.45,
                model=get_value_at_index(differentialdiffusion_171, 0),
                positive=get_value_at_index(cliptextencode_32, 0),
                negative=get_value_at_index(cliptextencode_4, 0),
                latent_image=get_value_at_index(ksampler_31, 0),
            )

            ksampler_45 = self.ksampler.sample(
                seed=seed,
                steps=4,
                cfg=1,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                denoise=get_value_at_index(primitive_float_crystools_97, 0),
                model=get_value_at_index(differentialdiffusion_171, 0),
                positive=get_value_at_index(cliptextencode_32, 0),
                negative=get_value_at_index(cliptextencode_4, 0),
                latent_image=get_value_at_index(ksampler_36, 0),
            )

            vaedecode_46 = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_45, 0),
                vae=get_value_at_index(self.checkpointloadersimple_1, 2),
            )

            impactgaussianblurmask_106 = self.impactgaussianblurmask.doit(
                kernel_size=20, sigma=10, mask=get_value_at_index(growmask_54, 0)
            )

            imageresizekj_178 = self.imageresizekj.resize(
                width=get_value_at_index(getimagesizeandcount_179, 1),
                height=get_value_at_index(getimagesizeandcount_179, 2),
                upscale_method="lanczos",
                keep_proportion=False,
                divisible_by=2,
                crop="disabled",
                image=get_value_at_index(vaedecode_46, 0),
            )
            for res in imageresizekj_178[0]:
                img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                img.save("save1.png")
                
            return img

if __name__ == "__main__":
    workflow = ObjectRemovalWorkflow()
    img = workflow()
    img.save("save2.png")
