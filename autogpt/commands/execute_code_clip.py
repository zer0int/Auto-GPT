"""Execute code in a Docker container"""
import os
import subprocess
from pathlib import Path

import docker
from docker.errors import ImageNotFound

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.visionconfig import visionhack

CFG = Config()

@command("run_clip", "Run CLIP", '"image_filename": "<image_filename>"')
def run_clip(image_filename: str) -> str:
    """Run the CLIPrun.py script with a given image filename

    Args:
        image_filename (str): The name of the image file

    Returns:
        str: The output of the CLIPrun.py script
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    workspace_directory = f"{visionhack}/auto_gpt_workspace"
    if str(workspace_directory) not in current_dir:
        os.chdir(workspace_directory)
    # Construct the full image path
    image_path = f"{visionhack}/images/{image_filename}"
    command_line = f"python CLIPrun.py --image_path {image_path}"
    print(f"Executing command 'run_clip' in working directory...")

    result = subprocess.run(command_line, capture_output=True, shell=True, encoding="utf-8")
    # Extract the output filename from the result.stdout
    output_filename_line = ""
    for line in result.stdout.split("\n"):
        if "CLIP tokens saved to" in line:
            output_filename_line = line
            break

    output = f"{output_filename_line}"

    # Change back to whatever the prior working dir was
    os.chdir(current_dir)

    return output

@command("run_shape", "Run SHAPE", '"prompt": "<prompt>"')
def run_shape(prompt: str) -> str:
    """Run the SHAPErun.py script with a given prompt

    Args:
        prompt (str): The text prompt to use for generating the 3D image

    Returns:
        str: The output of the SHAPErun.py script
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    workspace_directory = f"{visionhack}/auto_gpt_workspace"
    if str(workspace_directory) not in current_dir:
        os.chdir(workspace_directory)

    command_line = f'python SHAPErun.py --prompt "{prompt}"'
    #print(f"Executing command '{command_line}' in working directory '{os.getcwd()}'")
    print(f"Executing command 'run_shape' in working directory...")

    result = subprocess.run(command_line, capture_output=True, shell=True, encoding="utf-8")
    # Extract the output filename from the result.stdout
    output_filename_line = ""
    for line in result.stdout.split("\n"):
        if "SHAPE image" in line:
            output_filename_line = line
            break

    output = f"{output_filename_line}"

    # Change back to whatever the prior working dir was
    os.chdir(current_dir)

    return output