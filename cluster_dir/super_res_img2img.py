import torch
from PIL import Image
from io import BytesIO
import random
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

num_inference_steps = 100
guidance_scale = 0
strength_list = np.arange(0.1, 1, 0.05)
num_of_imgs = 3
upscale = 3
prompt_list = ["greyscale image of number 5"] * num_of_imgs

# loading image with PIL
img = Image.open("./mnist_dataset_train/0.png").convert("RGB")
# Trying to not convert to RGB
img = Image.open("./mnist_dataset_train/0.png")
img_h, img_w = img.size
imgReSi = img.resize((img_h * upscale, img_w * upscale), Image.Resampling.BICUBIC)

# load the pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id_or_path = "./stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16",
    torch_dtype=torch.float16,
).to(device)


# Removing NSFW
def dummy(images, **kwargs):
    return images, False

pipe.safety_checker = dummy

init_image = imgReSi

for strength in strength_list:
    for idx, prompt in enumerate(prompt_list):
        image = pipe(
            init_image=init_image,
            prompt=prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=None,  # Can also use the generator created above if a seed is needed
            guidance_scale=guidance_scale  # Higher guidance scale makes image more linked to text promt (less freedom)
        ).images[0]
        save_promt = prompt.replace(' ', '-').replace(',', '')
        image.save(f'imgs/{save_promt}_infsteps{num_inference_steps}_'
                   f'gs{guidance_scale}_s{strength}_{random.randint(0, 1e6)}.png')

