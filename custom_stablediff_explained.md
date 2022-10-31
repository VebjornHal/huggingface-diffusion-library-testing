# Explaination of custom_stablediff.py

## Setup of components, models and parameters

#### Imports 
```python

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
import numpy
from torchvision import transforms as tfms
import random

```

#### Path to local diffusion and tokenizer model
```python

model_path = "./stable-diffusion-v1-5"
tokenizer_path = "./clip-vit-large-patch14"

```

#### Loading different components of the models:
```python

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", use_auth_token=True)

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
text_encoder = CLIPTextModel.from_pretrained(tokenizer_path)

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", use_auth_token=True)


```

#### Creating a noise schedule for the diffusion process:
```python

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)

```

#### Putting all components of the model into GPU mode
```python

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

```


#### Creating different variables for parameters
```python

prompt = [
    "A digital illustration of a steampunk computer laboratory with clockwork machines, 4k, detailed, trending in "
    "artstation, fantasy vivid colors"]

height = 512
width = 512
num_inference_steps = 200
guidance_scale = 7.5
generator = torch.manual_seed(4)
batch_size = 1

```

## Preparations before the diffusion process

#### Tokenizing the text and encoding it

Essentially creating a dict of two tensors of the text which will be used to guide the diffusion process.
The first tensor is the input_ids and the second is the attention_mask. Input ids are the ids of the tokens.
Attention mask is a binary tensor which is 1 for tokens and 0 for padding tokens. It's used to avoid performing
attention on padding tokens.

```python

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                       return_tensors="pt")

```

* promt = The text which will be tokenized
* padding = max_length: Pad to a maximum length specified with the argument max_length or to the maximum acceptable 
input length for the model if that argument is not provided.
* truncation = True: Truncate to a maximum length specified with the argument max_length or to the maximum acceptable 
input length for the model if that argument is not provided. This will truncate token by token, 
removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
* return_tensors = "pt": Return PyTorch tensors.

#### Encoding the text with the text encoder 

```python

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

```
* with torch.no_grad(): This context manager can be used to disable gradient calculation within the block.
* input_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of input sequence tokens in the
vocabulary. Padding will be ignored by default should you provide it.
* Extracting input ids from text input by doing "text_input.input_ids" and putting it into the text encoder with gpu mode

Unconditional embedding is created in order to later have guidance scaling as an option. This is an embedding without
any input prompt. The last line concatenates both the unconditional and conditional embeddings.

#### Prepping scheduler

The scheduler containes all parameters for the gaussian distributions at each timestep $t$, such as $\alpha_t$, 
$\cumprod \alpha_t$ and $\beta_t$.

```python

scheduler.set_timesteps(num_inference_steps)

```

#### Prepping the latent space

Here random noise in latent space is created. A divisor of 8 is used to achieve lower dimensionality.
The strucutre of the latent space is (batch_size, 512, height/8, width/8). Also we put in the generator as parameter
in order to use the wanted seed. The two last lines make sure the latent space is ready for gpu and that latent space
scaled to match *k* (dont fully understand this line).

```python 

latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator)
latents = latents.to(torch_device)
latents = latents * scheduler.sigmas[0]  # Need to scale to match k

```

#### The diffusion loop, which is the main part of the script

Autocast serve as a context manager that allows regions of your script to run in mixed precision. This can often
result in significant speedups while retaining comparable accuracy. The main benefit of using autocast is that
you don’t have to manually manage the casts to/from float16/float32, which can be error-prone and hard to maintain.

```python
with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
```

The line "latent_model_input = torch.cat([latents] * 2)" is used to avoid doing two forward passes. This works since
the text_embeddings contain two different tensors, one with unconditional embedding and one with conditional embedding.\

#sigma = scheduler.sigmas[i] is used to get the sigma value for the current timestep. \

latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5) is the formula to for applying noise to the 
latent_mode_input. \

Next we predict the noise residual. This is done by using the unet model with the latent_model_input, 
the current timestep and the text_embeddings. \

Next ww predict the noise for the current time step (with torch.no_grad()) for both conditional and unconditional
text embeddings. \

Now we can split the unconditional and conditional noise predictions by using "noise_pred.chunk(2)". \

The next line takes into account the guidance scale:

```python

noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

```

If guidance scale < 0, the noise from the conditional and unconditional will be the same and noise_pred will be equal
to noise_pred_uncond. If guidance scale > 0, the noise from the conditional and unconditional will be different and
noise_pred will get the addition of guidance scaled multiplication of the difference. \

The last line: 

```python
 
latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

```

is used to compute the previous noisy sample x_t -> x_t-1. Then the loop start again with x_t-1 as input and does the same for the amount
steps specified in the scheduler. \

#### Scale and decode the image latents with vae

First there is a simple scaling formula. Next we utilize the vae model to decode the image latents. 

```python

latents = 1 / 0.18215 * latents

with torch.no_grad():
    image = vae.decode(latents)[0]
    
```


    













