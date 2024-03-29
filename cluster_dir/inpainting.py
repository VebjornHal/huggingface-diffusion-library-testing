from io import BytesIO
import torch
from torch import autocast
import requests
import PIL
import ftfy
import random
import argparse
from diffusers import StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler

# Important parameters which can be tweaked
num_inference_steps = 500
guidance_scale = 7.5
num_of_imgs = 5
img_h = 512
img_w = 512
useFP16 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-img", "--image", type=str, default="img2", help="Filename of image to be inpainted")

parser.add_argument("-mask", "--mask", type=str, default="mask2", help="Filename of mask image to be inpainted")

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

img_name = args.image
mask_name = args.mask
txt_prompt = args.prompt
num_of_imgs = args.num_of_imgs
guidance_scale = args.guidance_scale
img_h = args.height
img_w = args.width
num_inference_steps = args.num_inference_steps
strength = args.strength

# Checking if height and width is dividable by 8 and if not, make it dividable by 8
if img_h % 8 != 0 or img_w % 8 != 0:
    img_h = int(img_h / 8) * 8
    img_w = int(img_w / 8) * 8

# print variables
print("img_name: ", img_name)
print("mask_name: ", mask_name)
print("txt_prompt: ", txt_prompt)
print("num_of_imgs: ", num_of_imgs)
print("guidance_scale: ", guidance_scale)
print("img_h: ", img_h)
print("img_w: ", img_w)
print("num_inference_steps: ", num_inference_steps)
print("strength: ", strength)

# With resizing
# init_image = PIL.Image.open(f'./inpainting_imgs_test/{img_name}').convert("RGB").resize((512, 512))
# mask_image = PIL.Image.open(f'./inpainting_imgs_test/{mask_name}').convert("RGB").resize((512, 512))

# Trying to not resize the images to 512x512
init_image = PIL.Image.open(f'./inpainting_imgs_test/{img_name}').convert("RGB").resize((img_h, img_w))
mask_image = PIL.Image.open(f'./inpainting_imgs_test/{mask_name}').convert("RGB").resize((img_h, img_w))

prompt_list = [txt_prompt] * num_of_imgs

# Creating custom generator, does not have to be used
generator = torch.Generator(device=device).manual_seed(0)  # change the seed to get different results

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "./stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16).to(device)


# Dummy function to replace the safety checker function in order to turn of the faulty NSFW filter from huggingface
def dummy(images, **kwargs):
    return images, False


pipe.safety_checker = dummy

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

# The loop for generating and saving images with the use of the promt_list.
for idx, prompt in enumerate(prompt_list):
    image = pipe(prompt=prompt,
                 image=init_image,
                 mask_image=mask_image,
                 height=img_h,
                 width=img_w,
                 guidance_scale=guidance_scale,
                 generator=None,
                 num_inference_steps=num_inference_steps,
                 strength=strength,
                 ).images[0]
    save_promt = prompt.replace(' ', '-').replace(',', '')
    image.save(f'imgs/{save_promt}_nis{num_inference_steps}_gs{guidance_scale}_s{strength}'
               f'_{random.randint(0, 1e6)}.png')
