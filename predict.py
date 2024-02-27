# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File




from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from typing import Any

import requests
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64


def save_image(numpy_image, save_directory= "./tmp/out/", image_name= "img1.png"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Convert the NumPy image array to PIL image
    pil_image = Image.fromarray(np.uint8(numpy_image))
    
    # Construct the save path
    save_path = os.path.join(save_directory, image_name)
    
    # Save the image to disk
    pil_image.save(save_path)
    
    # Return the save path
    return save_path

def decode_base64_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)

def get_image_from_url(image_url):

    # Open the image using PIL (Python Imaging Library)
    pil_image = Image.open(image_url)
    
    # Convert the PIL image to a NumPy array
    np_image = np.array(pil_image)
    
    # Process the image as needed
    # Example: Print the shape of the NumPy array
    print("Image shape:", np_image.shape)

    return np_image

class Predictor(BasePredictor):
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")  
        self.apply_uniformer = UniformerDetector()
        # Get the current directory
        current_dir = os.getcwd()
        # os.path.dirname(os.path.abspath(__file__))

        # Construct the file path using the current directory and the relative path
        model_path = os.path.join(current_dir, 'models', 'cldm_v15.yaml')

        self.model = create_model(model_path).cuda()
        self.model.load_state_dict(load_state_dict(os.path.join(current_dir, 'models', 'control_sd15_seg.pth'), location='cuda'))
        self.model =self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def predict(self,input_image_string: str = Input(description="Image to process") , prompt: str =  Input(description= "prompt"),
                a_prompt: str = Input(description= "added prompt", default="best quality, extremely detailed"),
                n_prompt: str = Input(description= "negative prompt", default= "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
                num_samples: int = Input(description= "num_samples", default= 1), image_resolution: int = Input(description= "Image resolution", default= 512),
                detect_resolution: int = Input(description= "detect_resolution", default= 512),
                ddim_steps: int = Input(description= "ddim steps", default=20), guess_mode: bool = Input(description= "guess mode", default=False),
                strength: float = Input(description= "strength", default=1.0), scale: float = Input(description= "scale", default=9.0), 
                seed: float = Input(description= "seed", default=3.5), eta: float = Input(description= "eta", default=0.0),
                ) -> Any:
        
        
        
        with torch.no_grad():
            input_image= decode_base64_image(input_image_string)
            input_image = HWC3(input_image)
            detected_map = self.apply_uniformer(resize_image(input_image, detect_resolution))
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
               self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
               self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
               self.model.low_vram_shift(is_diffusing=False)

            x_samples =self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            output_paths = []
            for i, sample in enumerate(results):
                output_path = f"/tmp/out-{i}.png"
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                print(type(sample))
                print(sample.shape)
                cv2.imwrite(output_path, sample)
                # sample.save(output_path)
                output_paths.append(Path(output_path))
        
        # final_result= [detected_map] + results
        # result_path= save_image(final_result)
        
        return output_paths

