# -*- coding: utf-8 -*-
"""CLIP GradCAM Visualization for AutoGPT

2023 GPT-4 & zer0int -- Twitter: @zer0int1

Adapted for AutoGPT; takes /AutoGPT/images/0001.png + the CLIP opinion tokens about this image from /Auto-GPT/auto_gpt_workspace/tokens_0001.txt 
Uses the CLIP tokens as captions, and computes the heatmap for them one-by-one in just a few seconds, dumping the heatmap-overlay images all in /Auto-GPT/GradCAM.

Basically a debug CLIP for humans, i.e. "What the heck is CLIP 'looking' at?!".


Originally obtained from https://github.com/kevinzakka/clip_playground

# CLIP GradCAM Colab

This Colab notebook uses [GradCAM](https://arxiv.org/abs/1610.02391) on OpenAI's [CLIP](https://openai.com/blog/clip/) model to produce a heatmap highlighting which regions in an image activate the most to a given caption.

**Note:** Currently only works with the ResNet variants of CLIP. ViT support coming soon.
"""

#@title Install dependencies

#@markdown Please execute this cell by pressing the _Play_ button 
#@markdown on the left.

#@markdown **Note**: This installs the software on the Colab 
#@markdown notebook in the cloud and not on your computer.

#!pip install ftfy regex tqdm matplotlib opencv-python scipy scikit-image
#!pip install git+https://github.com/openai/CLIP.git

import os
import sys
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from PIL import ImageDraw, ImageFont
import argparse
ffs = os.getcwd()
print(ffs)
sys.path.append(".././")
ffs = os.getcwd()
print(ffs)
from autogpt.visionconfig import visionhack

parser = argparse.ArgumentParser(description="Image to use (from Auto-GPT/images folder), Tokens txt to use (from Auto-GPT/auto_gpt_workspace folder)")
parser.add_argument("--image", type=str, required=True, default="0001.png", help="Input image")
parser.add_argument("--txt", type=str, required=True, default="tokens_0001.txt", help="Input image")
args = parser.parse_args()

#@title Helper functions

#@markdown Some helper functions for overlaying heatmaps on top
#@markdown of images and visualizing with matplotlib.

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn_single(image_np, attn_map, blur=True, image_name="0001.png", model_name="RN50x4", layer_name="layer1", caption="AI"):
    attn_map_img = getAttMap(image_np, attn_map, blur)
    attn_map_pil = Image.fromarray((attn_map_img * 255).astype(np.uint8))
    
    draw = ImageDraw.Draw(attn_map_pil)
    font = ImageFont.load_default()

    text = f"{image_name}\n{model_name} {layer_name}\n{caption}"
    text_width, text_height = draw.textsize(text, font)
    draw.multiline_text((10, 10), text, fill=(255, 255, 255), font=font)
    
    #plt.imshow(attn_map_pil)
    #plt.axis("off")
    #plt.show()

    return attn_map_pil
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

#@title GradCAM: Gradient-weighted Class Activation Mapping

#@markdown Our gradCAM implementation registers a forward hook
#@markdown on the model at the specified layer. This allows us
#@markdown to save the intermediate activations and gradients
#@markdown at that layer.

#@markdown To visualize which parts of the image activate for
#@markdown a given caption, we use the caption as the target
#@markdown label and backprop through the network using the
#@markdown image as the input.
#@markdown In the case of CLIP models with resnet encoders,
#@markdown we save the activation and gradients at the
#@markdown layer before the attention pool, i.e., layer4.

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

#@title Run

#@markdown #### Image & Caption settings
#image_url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_706,w_1256,x_0,y_64/f_auto,q_auto,w_1100/v1554995050/shape/mentalfloss/516438-istock-637689912.jpg' #@param {type:"string"}

#image_caption = '' #@param {type:"string"}
#@markdown ---
#@markdown #### CLIP model settings
clip_model = "RN50x4" #@param ["ViT-B/16", "ViT-B/32", "ViT-L/14", "RN50x16", "RN50x4","RN50","RN101","RN50x64"]
#saliency_layer = "layer1" #@param ["layer4", "layer3", "layer2", "layer1"]
#@markdown ---
#@markdown #### Visualization settings
blur = True #@param {type:"boolean"}

import imageio
from skimage import *
import skimage.io

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device, jit=False)

saliency_layer = ''
image_caption = ''

gradimg = args.image
# Image that was used for the "CLIP opinion" tokens
image_path = f"{visionhack}/images/{gradimg}"
#urllib.request.urlretrieve(image_url, image_path)
# channel-2478x.png

image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
image_np = load_image(image_path, model.visual.input_resolution)
text_input = clip.tokenize([image_caption]).to(device)

gradtxt = args.txt
# Read the text file containing CLIP opinion tokens and split the words
text_file_path = f"{visionhack}/auto_gpt_workspace/{gradtxt}"
with open(text_file_path, "r") as f:
    words = f.read().strip().split(" ")

# Add an empty string to the beginning of the words list to represent an empty image_caption
words.insert(0, "")

saliency_layers = ["layer1", "layer2", "layer3", "layer4"]

# Iterate through each word in the text file
for image_caption in words:
    # Iterate through each saliency layer
    for saliency_layer in saliency_layers:
        attn_map = gradCAM(
            model.visual,
            image_input,
            model.encode_text(clip.tokenize([image_caption]).to(device)).float(),
            getattr(model.visual, saliency_layer)
        )

        attn_map = attn_map.squeeze().detach().cpu().numpy()

        # Call the modified viz_attn_single function and save the image
        attn_image = viz_attn_single(image_np, attn_map, blur, image_name="0001.png", model_name="RN50x4", layer_name=saliency_layer, caption=image_caption)

        # Prepare the file name
        image_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
        layer_name_short = saliency_layer.replace("layer", "L")
        formatted_file_name = f"{image_name_without_extension}-{clip_model}_{layer_name_short}-{image_caption}.png"

        # Save the GradCAM image with text
        save_path = os.path.join(f"{visionhack}/GradCAM", formatted_file_name)
        attn_image.save(save_path)

        print(f"GradCAM image saved to: {save_path}")


from PIL import Image
import math

# Compute the number of rows and columns for the final image
num_rows = len(words)
num_columns = len(saliency_layers)


from pathlib import Path

gradcam_folder = Path(f"{visionhack}/GradCAM")
gradcam_images = list(gradcam_folder.glob("*.png"))  # Get list of all .png files in the folder

if gradcam_images:  # Check if the list is not empty
    sample_image_path = gradcam_images[0]  # Take the first image as the sample
    sample_image = Image.open(sample_image_path)
    width, height = sample_image.size
else:
    print("No GradCAM images found.")
    sys.exit(1)

# Calculate the number of final images to be created
max_rows_per_image = 5
num_final_images = math.ceil(num_rows / max_rows_per_image)

# Create the final images
for image_index in range(num_final_images):
    # Create a new blank image to paste all the GradCAM images onto
    final_image = Image.new("RGB", (width * num_columns, height * max_rows_per_image))

    # Iterate through the words and saliency layers, pasting the corresponding GradCAM images
    for row in range(max_rows_per_image):
        word_index = image_index * max_rows_per_image + row
        if word_index >= num_rows:
            break
        image_caption = words[word_index]
        for col, saliency_layer in enumerate(saliency_layers):
            layer_name_short = saliency_layer.replace("layer", "L")
            file_name = f"{image_name_without_extension}-{clip_model}_{layer_name_short}-{image_caption}.png"
            image_path = os.path.join(f"{visionhack}/GradCAM", file_name)
            gradcam_image = Image.open(image_path)
            final_image.paste(gradcam_image, (width * col, height * row))

    # Save the final image
    final_image_name = f"ALL_{image_index + 1}_{image_name_without_extension}-{clip_model}.png"
    final_image_path = os.path.join(f"{visionhack}/GradCAM", final_image_name)
    final_image.save(final_image_path)

    print(f"Final image {image_index + 1} saved to: {final_image_path}")

