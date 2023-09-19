import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
from PIL import Image

model_id = "timbrooks/instruct-pix2pix"
DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, cache_dir="./weights")
pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
pipe.to("cuda")
pipe.scheduler.set_timesteps(100)
image = Image.open("data/tandt/truck/images/000001.jpg")
images = pipe("Make the tree leaves yellow", image=image).images
images[0].show()
