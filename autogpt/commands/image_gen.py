""" Image Generation Module for AutoGPT."""
import io
import os
import sys
import uuid
import shutil
import subprocess
from base64 import b64decode
from pathlib import Path

import openai
import requests
from PIL import Image
import re

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.visionconfig import visionhack
from autogpt.commands.execute_code import execute_shell

CFG = Config()

def is_relative_to(path, *other):
    try:
        path.relative_to(*other)
        return True
    except ValueError:
        return False

@command("generate_image", "Generate Image", '"prompt": "<prompt>"', CFG.image_provider)
def generate_image(prompt: str, size: int = 256) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use
        size (int, optional): The size of the image. Defaults to 256. (Not supported by HuggingFace)

    Returns:
        str: The filename of the image
    """
    #workspace_directory = f"{visionhack}/auto_gpt_workspace"
    #f"python {Path(filename).relative_to(CFG.workspace_directory)}",
    workspace_path= "{visionhack}/auto_gpt_workspace"
    workspace_directory = "{visionhack}/images"
    #filename = f"{CFG.workspace_path}/{str(uuid.uuid4())}.jpg"
    filename = f"{workspace_path}/{str(uuid.uuid4())}.jpg"
    if is_relative_to(Path(filename), workspace_directory):
        relative_path = Path(filename).relative_to(workspace_directory)
        command_line = f'python {relative_path}'
    else:
        command_line = f'python {filename}'

    # DALL-E
    if CFG.image_provider == "dalle":
        return generate_image_with_dalle(prompt, filename, size)
    # HuggingFace
    elif CFG.image_provider == "huggingface":
        return generate_image_with_hf(prompt, filename)
    # SD WebUI
    elif CFG.image_provider == "sdwebui":
        return generate_image_with_sd_webui(prompt, filename, size)
    # stablediffusion
    elif CFG.image_provider == "stablediffusion":
        return generate_image_with_stablediffusion(prompt, size)
    return "No Image Provider Set"


def generate_image_with_hf(prompt: str, filename: str) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    API_URL = (
        f"https://api-inference.huggingface.co/models/{CFG.huggingface_image_model}"
    )
    if CFG.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {
        "Authorization": f"Bearer {CFG.huggingface_api_token}",
        "X-Use-Cache": "false",
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
        },
    )

    image = Image.open(io.BytesIO(response.content))
    logger.info(f"Image Generated for prompt:{prompt}")

    image.save(filename)

    return f"Saved to disk:{filename}"


def generate_image_with_dalle(prompt: str, filename: str, size: int) -> str:
    """Generate an image with DALL-E.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int): The size of the image

    Returns:
        str: The filename of the image
    """

    # Check for supported image sizes
    if size not in [256, 512, 1024]:
        closest = min([256, 512, 1024], key=lambda x: abs(x - size))
        logger.info(
            f"DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. Setting to {closest}, was {size}."
        )
        size = closest

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=f"{size}x{size}",
        response_format="b64_json",
        api_key=CFG.openai_api_key,
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    image_data = b64decode(response["data"][0]["b64_json"])

    with open(filename, mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk:{filename}"


def generate_image_with_sd_webui(
    prompt: str,
    filename: str,
    size: int = 512,
    negative_prompt: str = "",
    extra: dict = {},
) -> str:
    """Generate an image with Stable Diffusion webui.
    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int, optional): The size of the image. Defaults to 256.
        negative_prompt (str, optional): The negative prompt to use. Defaults to "".
        extra (dict, optional): Extra parameters to pass to the API. Defaults to {}.
    Returns:
        str: The filename of the image
    """
    # Create a session and set the basic auth if needed
    s = requests.Session()
    if CFG.sd_webui_auth:
        username, password = CFG.sd_webui_auth.split(":")
        s.auth = (username, password or "")

    # Generate the images
    response = requests.post(
        f"{CFG.sd_webui_url}/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_index": "DDIM",
            "steps": 20,
            "cfg_scale": 7.0,
            "width": size,
            "height": size,
            "n_iter": 1,
            **extra,
        },
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    # Save the image to disk
    response = response.json()
    b64 = b64decode(response["images"][0].split(",", 1)[0])
    image = Image.open(io.BytesIO(b64))
    image.save(filename)

    return f"Saved to disk:{filename}"

def generate_image_with_stablediffusion(prompt: str, size: int = 768) -> str:
    """Generate an image with Stable Diffusion.

    Args:
        prompt (str): The prompt to use

    Returns:
        str: The filename of the image
    """
    # Execute the stablediffusion.py script with the provided prompt
    command_line = f'python stablediffusion.py --prompt "{prompt}"'
    output = execute_shell(command_line)

    # Check for errors in the output
    if "Error" in output:
        return f"Error generating image: {output}"

    output_directory = f"{stablehome}/outputs/txt2img-samples/samples"
    workspace_directory = f"{visionhack}/images"

    # Find the highest numbered PNG file
    pattern = re.compile(r'^(\d{5})\.png$')
    max_number = -1
    filename = None

    for entry in os.listdir(output_directory):
        match = pattern.match(entry)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                filename = entry

    if filename:
        # Copy the file to the workspace directory
        generated_image_path = os.path.join(output_directory, filename)
        new_image_path = os.path.join(workspace_directory, filename)
        shutil.copyfile(generated_image_path, new_image_path)

        return f"stablediffusion generated image saved to {filename}"
    else:
        return "Error: Generated image not found."