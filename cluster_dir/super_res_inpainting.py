from io import BytesIO
import torch
from torch import autocast
import requests
import PIL
import ftfy
import random
import argparse
from diffusers import StableDiffusionInpaintPipelineLegacy, StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler
import numpy as np

# Important parameters which can be tweaked
num_inference_steps = 500
guidance_scale = 7
strength = 0
num_of_imgs = 10
upscale = 2
new_model = True

# img_h = 512
# img_w = 512


# txt_prompt = "white number 5 in middle of image, black background, greyscale image, black and white"
txt_prompt = "ice coffee standing on table"
useFP16 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = PIL.Image.open("inpainting_imgs_test/img2.png")


# Create a function for downscaling with PIL and make sure new width and height don't exceed 200 but keep same ratio
# as original image
def downscale(image, scale=1):
    img_w, img_h = image.size
    new_w = int(img_w / scale)
    new_h = int(img_h / scale)
    if new_w > 200:
        new_w = 200
        new_h = int(img_h / (img_w / new_w))
    elif new_h > 200:
        new_h = 200
        new_w = int(img_w / (img_h / new_h))
    return image.resize((new_w, new_h), PIL.Image.BICUBIC)


# Create a function for upscaling an image with replacing the pixels and have value zero in between
def image_explode(image, scale):
    img_array = np.array(image).transpose(2, 0, 1)
    c, h, w = img_array.shape
    # Check if image contain alpha channel and if it does remove it
    if c == 4:
        img_array = img_array[:3]
        c, h, w = img_array.shape
    # Create a new array with zeros
    new_img_array = np.zeros((c, h * scale, w * scale))
    # Fill the new array with the values from the original array
    new_img_array[:, ::scale, ::scale] = img_array[:, :, :]
    new_c, new_h, new_w = new_img_array.shape
    # Create a mask image which fills every odd row and column with value 255 (white)
    mask = np.zeros((new_c, new_h, new_w))
    for i in range(1, new_w, scale):
        mask[:, :, i:(i + scale - 1)] = 255
    for i in range(1, new_h, scale):
        mask[:, i:(i + scale - 1), :] = 255

    # Return the new array
    new_img_array = PIL.Image.fromarray(new_img_array.transpose(1, 2, 0).astype(np.uint8))
    mask = PIL.Image.fromarray(mask.transpose(1, 2, 0).astype(np.uint8))
    return new_img_array, mask


def advanced_image_explode(image, scale):
    img_array = np.array(image).transpose(2, 0, 1)
    c, h, w = img_array.shape
    # Check if image contain alpha channel and if it does remove it
    if c == 4:
        img_array = img_array[:3]
        c, h, w = img_array.shape
    # Create a new array with zeros
    new_img_array = np.zeros((c, h * scale, w * scale))
    # Fill the new array with the values from the original array
    new_img_array[:, ::scale, ::scale] = img_array[:, :, :]

    # Move all even numbered rows and columns to the right and down
    new_img_array[:, ::scale, 1::scale] += img_array[:, :, :]
    new_img_array[:, 1::scale, ::scale] += img_array[:, :, :]
    new_img_array[:, 1::scale, 1::scale] += img_array[:, :, :]

    new_c, new_h, new_w = new_img_array.shape
    # Create a mask image which fills every odd row and column with value 255 (white)
    mask = np.zeros((new_c, new_h, new_w))
    for i in range(1, new_w, scale + 1):
        mask[:, :, i:(i + scale)] = 255
    for i in range(1, new_h, scale + 1):
        mask[:, i:(i + scale), :] = 255

    # Return the new array
    new_img_array = PIL.Image.fromarray(new_img_array.transpose(1, 2, 0).astype(np.uint8))
    mask = PIL.Image.fromarray(mask.transpose(1, 2, 0).astype(np.uint8))
    return new_img_array, mask


downscaled_img = downscale(img)

init_image, mask_image = advanced_image_explode(downscaled_img, upscale)

downscaled_img.show()
init_image.show()
mask_image.show()

# Setting hight and width
img_w, img_h = init_image.size

# Checking if height and width is dividable by 8 and if not, make it dividable by 8
if img_h % 8 != 0 or img_w % 8 != 0:
    img_h = int(img_h / 8) * 8
    img_w = int(img_w / 8) * 8

prompt_list = [txt_prompt] * num_of_imgs

# Creating custom generator, does not have to be used
generator = torch.Generator(device=device).manual_seed(0)  # change the seed to get different results

if new_model:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16).to(device)
else:
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        "./stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16).to(device)


# Dummy function to replace the safety checker function in order to turn of the faulty NSFW filter from huggingface
def dummy(images, **kwargs):
    return images, False


pipe.safety_checker = dummy

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

print("txt_prompt: ", txt_prompt)
print("num_of_imgs: ", num_of_imgs)
print("guidance_scale: ", guidance_scale)
print("img_h: ", img_h)
print("img_w: ", img_w)
print("num_inference_steps: ", num_inference_steps)
print("strength: ", strength)

# The loop for generating and saving images with the use of the promt_list.

if new_model:
    for idx, prompt in enumerate(prompt_list):
        image = pipe(prompt=prompt,
                     image=init_image,
                     mask_image=mask_image,
                     height=img_h,
                     width=img_w,
                     guidance_scale=guidance_scale,
                     generator=None,
                     num_inference_steps=num_inference_steps,
                     ).images[0]
        save_promt = prompt.replace(' ', '-').replace(',', '')
        image.save(f'imgs/{save_promt}_nis{num_inference_steps}_gs{guidance_scale}_s{strength}'
                   f'_{random.randint(0, 1e6)}.png')
else:
    for idx, prompt in enumerate(prompt_list):
        image = pipe(prompt=prompt,
                     init_image=init_image,
                     mask_image=mask_image,
                     height=img_h,
                     width=img_w,
                     guidance_scale=guidance_scale,
                     generator=None,
                     num_inference_steps=num_inference_steps,
                     strength=strength
                     ).images[0]
        save_promt = prompt.replace(' ', '-').replace(',', '')
        image.save(f'imgs/{save_promt}_nis{num_inference_steps}_gs{guidance_scale}_s{strength}'
                   f'_{random.randint(0, 1e6)}.png')
