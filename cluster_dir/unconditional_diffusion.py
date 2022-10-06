from torch import autocast
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "./ddpm-celebahq-256"
device = "cuda"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to(device)

# run pipeline in inference (sample random noise and denoise)
with autocast("cuda"):
    image = ddpm().images[0]

# save image
image.save("imgs/ddpm_generated_image.png")