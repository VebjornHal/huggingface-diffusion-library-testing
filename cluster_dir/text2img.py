import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import random
import argparse

num_inference_steps = 500
guidance_scale = 7.5
num_of_imgs = 5
img_h = 512
img_w = 512

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prompt', metavar='prompt', type=str, help='enter text prompt',
                    default='a home for all the critters of the forest,'
                            ' big tree, tall, lush, calm, book cover, ultra realistic, 4k, 8k')
parser.add_argument('-n', '--num_of_imgs', metavar='number of images', type=int,
                    help='enter desired number of images', default=5)
parser.add_argument('-gs', '--guidance_scale', metavar='guidance scale', type=float,
                    help='enter number between ', default=7.5)
parser.add_argument('-H', '--height', metavar='height', type=int,
                    help=('height of generated image in number of pixels'),
                    default=512)
parser.add_argument('-W', '--width', metavar='width', type=int,
                    help=('width of generated image in number of pixels'),
                    default=512)
parser.add_argument('-n-inf-s', '--num_inference_steps', metavar='number of inference steps',
                    help='enter number of inference steps', type=int, default=70)
args = parser.parse_args()

txt_prompt = args.prompt
num_of_imgs = args.num_of_imgs
guidance_scale = args.guidance_scale
img_h = args.height
img_w = args.width
num_inference_steps = args.num_inference_steps

# Checking if height and width is dividable by 8 and if not, make it dividable by 8
if img_h % 8 != 0 or img_w % 8 != 0:
    img_h = int(img_h / 8) * 8
    img_w = int(img_w / 8) * 8

# Printing all variables to console
print('txt_prompt: ', txt_prompt)
print('num_of_imgs: ', num_of_imgs)
print('guidance_scale: ', guidance_scale)
print('img_h: ', img_h)
print('img_w: ', img_w)
print('num_inference_steps: ', num_inference_steps)

# Creating pipe
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5",
                                               revision="fp16",
                                               torch_dtype=torch.float16
                                               )
pipe = pipe.to("cuda")

# Turning of NSFW filter by replacing with dummy function
# Dummy function to replace the safety checker function in order to turn of the faulty NSFW filter from huggingface
def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy

######################################--Example promts--############################################################
# prompt_list = ['fantasy land, in style of adventure time',
#                'dog eating cat',
#                'character portrait, cute child holding a butchersknife, 1 ,in a highly detailed toystore, ultra detail',
#                'a city from the future. Surrealism. in a surrealist style. Art by Artgerm and Greg Rutkowski, Bastien']

# prompt_list = ['hobbits house, 16k, 8k, hyper sharp focus, super resolution, stunning intricate detail, photo realistic',
#                'a painting of a fox sitting in a field at sunrise in the style of Claude Monet',
#                'A computer from the 90s in the style of vaporwave',
#                'A sea otter with a pearl earring" by Johannes Vermeer',
#                'A photo of a teddy bear on a skateboard in Times Square',
#                'a home for all the critters of the forest, big tree, tall , lush , calm , book cover , ultra realistic , 4k , 8k']
####################################################################################################################


prompt_list = [txt_prompt] * num_of_imgs

# Creating a generator mainly for setting some seed if needed, if seed is not uesed a new image will be created everytime using the same prompt
generator = torch.Generator(device="cuda").manual_seed(1024)

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

for idx, prompt in enumerate(prompt_list):
    with autocast("cuda"):
        image = pipe(height=img_h,  # Image height
                     width=img_w,  # Image width
                     prompt=prompt,
                     num_inference_steps=num_inference_steps,
                     generator=None,  # Can also use the generator created above if a seed is needed
                     guidance_scale=guidance_scale  # Higher guidance scale makes image more linked to text promt (less freedom)
                     ).images[0]
    save_promt = prompt.replace(' ', '-').replace(',', '')
    image.save(f'imgs/{save_promt}_infsteps{num_inference_steps}_'
               f'gs{guidance_scale}_{random.randint(0, 1e6)}.png')
