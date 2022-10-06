import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import random

# Setting up pipeline
######################################## OLD
# pipe = StableDiffusionPipeline.from_pretrained(
#     "./stable-diffusion-v1-4",
#     revision="fp16",
#     torch_dtype=torch.float16,
#     use_auth_token=access_token
# )
# pipe = pipe.to("cuda")
########################################

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4",
                                               revision="fp16",
                                               torch_dtype=torch.float16
                                               )
pipe = pipe.to("cuda")

# Checking number of gpus available
print('før')
print(torch.cuda.device_count())
print('etter')

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

prompt_list = ['a home for all the critters of the forest, big tree, tall , lush , calm , book cover , ultra realistic , 4k , 8k'] * 5

# Creating a generator mainly for setting some seed if needed, if seed is not uesed a new image will be created everytime using the same prompt
generator = torch.Generator(device="cuda").manual_seed(1024)

# Can be used to save memory
print(pipe.unet.config.attention_head_dim)
pipe.enable_attention_slicing(8)

num_inference_steps = 500
guidance_scale = 7.5

for idx, prompt in enumerate(prompt_list):
    with autocast("cuda"):
        image = pipe(height=1024,  # Image height
                     width=1024,  # Image width
                     prompt=prompt,
                     num_inference_steps=num_inference_steps,
                     generator=None,  # Can also use the generator created above if a seed is needed
                     guidance_scale=guidance_scale  # Higher guidance scale makes image more linked to text promt (less freedom)
                     ).images[0]
    save_promt = prompt.replace(' ', '-').replace(',', '')
    image.save(f'imgs/{save_promt}_infsteps{num_inference_steps}_'
               f'gs{guidance_scale}_{random.randint(0, 1e6)}.png')
