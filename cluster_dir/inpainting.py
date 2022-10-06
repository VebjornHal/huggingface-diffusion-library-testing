from io import BytesIO
import torch
from torch import autocast
import requests
import PIL
import ftfy
import random

from diffusers import StableDiffusionInpaintPipeline

init_image = PIL.Image.open('inpainting_imgs_test/img2.png').convert("RGB").resize((512, 512))
mask_image = PIL.Image.open('inpainting_imgs_test/mask2.png').convert("RGB").resize((512, 512))

# Creating the inpainting pipline
device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained("./stable-diffusion-v1-4",
                                                      revision="fp16",
                                                      torch_dtype=torch.float16,
                                                      use_auth_token=False
                                                      ).to(device)


# Dummy function to replace the safety checker function in order to turn of the faulty NSFW filter from huggingface
def dummy(images, **kwargs):
    return images, False


# Turning of NSFW filter by replacing with dummy function
pipe.safety_checker = dummy

num_imgs = 10
prompt_list = ["thrusting rocket flying to space, super realistic, 4k"] * num_imgs

# Creating custom generator, does not have to be used
generator = torch.Generator(device="cuda").manual_seed(123123456)

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

num_inference_steps = 400
guidance_scale = 5.0
strength = 0.8

# The loop for generating and saving images with the use of the promt_list.
for idx, prompt in enumerate(prompt_list):
    with autocast("cuda"):
        image = pipe(prompt=prompt,
                     init_image=init_image,
                     mask_image=mask_image,
                     strength=strength,  # Default 0.8
                     guidance_scale=guidance_scale,  # Default 7.5
                     num_inference_steps=num_inference_steps,
                     generator=None).images[0]
    save_promt = prompt.replace(' ', '-').replace(',', '')
    image.save(f'imgs/{save_promt}_nis{num_inference_steps}_gs{guidance_scale}_s{strength}'
               f'_{random.randint(0, 1e6)}.png')
