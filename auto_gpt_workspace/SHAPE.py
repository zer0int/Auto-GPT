# -*- coding: utf-8 -*-
"""

2023, GPT-4 & zer0int / Twitter: @zer0int1

Adaptation of the original notebook sample_text_to_3d.ipynb / Shap-E by OpenAI, 2023 https://github.com/openai/shap-e

"""

import os
import sys
import torch
import re
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget #from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
import argparse
parser = argparse.ArgumentParser(description="Generate 3D images from text using GPT-4")
parser.add_argument('--prompt', type=str, required=True, help='The text prompt to use for generating the 3D mesh and image')
parser.add_argument('--output-folder', type=str, default=".././images", help='The folder to save the generated images')
args = parser.parse_args()
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = 15.0
prompt = args.prompt

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', filename)

output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

def get_next_filename(output_folder):
    existing_files = glob.glob(f"{output_folder}/s_*.png")
    if not existing_files:
        return "s_000.png"
    existing_files.sort()
    max_file = existing_files[-1]
    max_file_number = int(re.findall(r'\d+', max_file)[0])# 0 is overwrite, -1 is next iterate
    #print(f"{max_file_number}")
    next_file_number = max_file_number + 1
    #print(f"{next_file_number}")
    #save_diz = f"s_{next_file_number:03}.png"
    return f"s_{next_file_number:03}.png"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,#64
    sigma_min=1e-3,
    sigma_max=160,#noise level, potentially need to increase karras
    s_churn=0,#probably rate of turnover for new elements to be probes
)

render_mode = 'stf' # you can change this to 'stf' or 'nerf'
size = 512 # 64  # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    from PIL import Image
    import io
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    # Create a gif file from the images list
    with io.BytesIO() as buffer:
        images[0].save(
            buffer, format='PNG', save_all=False, append_images=images[1:], duration=1, loop=0
        )
        gif_data = buffer.getvalue()
        png_filename = get_next_filename(output_folder) # result is overwrite file
        #png_filename = next_file_number(output_folder)
        png_filepath = os.path.join(output_folder, png_filename)
        with open(png_filepath, 'wb') as f:
          f.write(gif_data)
        print(png_filename)

from shap_e.util.notebooks import decode_latent_mesh
import re

for i, latent in enumerate(latents):
    sanitized_prompt = sanitize_filename(prompt)
    ply_filename = f"{sanitized_prompt}_{i}.ply"
    ply_filepath = os.path.join(output_folder, ply_filename)
    with open(ply_filepath, 'wb') as f:
        decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)