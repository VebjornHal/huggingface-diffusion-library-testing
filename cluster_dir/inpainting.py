from io import BytesIO
import torch
from torch import autocast
import requests
import PIL
import ftfy
import random
import argparse

num_inference_steps = 500
guidance_scale = 7.5
num_of_imgs = 5
img_h = 512
img_w = 512

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prompt', metavar='prompt', type=str, help='enter text prompt',
                    default='cola can, red, standing on wooden table, 4k, 8k')
parser.add_argument('-n', '--num_of_imgs', metavar='number of images', type=int,
                    help='enter desired number of images', default=5)
parser.add_argument('-gs', '--guidance_scale', metavar='guidance scale', type=float,
                    help='enter number above 0', default=7.5)
parser.add_argument('-H', '--height', metavar='height', type=int,
                    help=('height of generated image in number of pixels'),
                    default=512)
parser.add_argument('-W', '--width', metavar='width', type=int,
                    help=('width of generated image in number of pixels'),
                    default=512)
parser.add_argument('-n-inf-s', '--num_inference_steps', metavar='number of inference steps',
                    help='enter number of inference steps', type=int, default=70)
parser.add_argument('-s', '--strength', metavar='strength', help='type in desired strength between 0 and 1)',
                    type=float, default=0.7)
args = parser.parse_args()

txt_prompt = args.prompt
num_of_imgs = args.num_of_imgs
guidance_scale = args.guidance_scale
img_h = args.height
img_w = args.width
num_inference_steps = args.num_inference_steps
strength = args.strength

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

prompt_list = [txt_prompt] * num_of_imgs

# Creating custom generator, does not have to be used
generator = torch.Generator(device="cuda").manual_seed(123123456)

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

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




