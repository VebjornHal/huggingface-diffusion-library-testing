# huggingface-diffusion-library-testing
Using huggingface public diffusion library to generate images on the springfield cluster. 

## How to generate images from text on the cluster

First we have to transfer all necesarry files to the cluster storage space. This can be done with:

```

sh sync_to_cluster.sh

```

This will create a folder in the root directory of the cluster storage space called *diffusion_lib_test*

### **text2img**

In order to generate image from text promt with the cluster:

```

frink run inpainting.yaml -f 

```

For changing the image promt one can just change the variable *promt_list* in text2img.py.
Below is an example promt list: 

```python

prompt_list = ['a home for all the critters of the forest, big tree, tall , lush , calm , book cover , ultra realistic , 4k , 8k'] * num_of_imgs


```

### **Inpainting**

In order to use inpainting we have to place an image and a mask image into the folder cluster_dir/inpainting_imgs_test
Here is an example of such images: 

<p float="left">
  <img src="./cluster_dir/inpainting_imgs_test/img2.png" width=302 height=403>
  <img src="./cluster_dir/inpainting_imgs_test/mask2.png" width=302 height=403>
</p>

```

frink run inpainting.yaml -f 

```

In the file inpainting.py one can change the variable *prompt_list* in order to change the prompt.

### Changing the parameters

in the *.py* files where there are pipeline one can easily change the parameters:

* num_inference_steps: The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
* strength: Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
* guidance_scale: Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.


```python

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

```


