"""Execute code in a Docker container"""
import os
import sys
import re
import subprocess
from pathlib import Path

import docker
from docker.errors import ImageNotFound
from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.visionconfig import visionhack, stablehome

CFG = Config()


@command("execute_python_file", "Execute Python File", '"filename": "<filename>"')
def execute_python_file(filename: str) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (str): The name of the file to execute

    Returns:
        str: The output of the file
    """
    logger.info(f"Executing file '{filename}'")

    if not filename.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    if not os.path.isfile(filename):
        return f"Error: File '{filename}' does not exist."

    if we_are_running_in_a_docker_container():
        result = subprocess.run(
            f"python {filename}", capture_output=True, encoding="utf8", shell=True
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    try:
        client = docker.from_env()
        # You can replace this with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        image_name = "python:3-alpine"
        try:
            client.images.get(image_name)
            logger.warn(f"Image '{image_name}' found locally")
        except ImageNotFound:
            logger.info(
                f"Image '{image_name}' not found locally, pulling from Docker Hub"
            )
            # Use the low-level API to stream the pull response
            low_level_client = docker.APIClient()
            for line in low_level_client.pull(image_name, stream=True, decode=True):
                # Print the status and progress, if available
                status = line.get("status")
                progress = line.get("progress")
                if status and progress:
                    logger.info(f"{status}: {progress}")
                elif status:
                    logger.info(status)
        container = client.containers.run(
            image_name,
            f"python {Path(filename).relative_to(CFG.workspace_directory)}",
            volumes={
                CFG.workspace_directory: {
                    "bind": "/workspace",
                    "mode": "ro",
                }
            },
            working_dir="/workspace",
            stderr=True,
            stdout=True,
            detach=True,
        )

        container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs

    except docker.errors.DockerException as e:
        logger.warn(
            "Could not run the script in a container. If you haven't already, please install Docker https://docs.docker.com/get-docker/"
        )
        return f"Error: {str(e)}"

    except Exception as e:
        return f"Error: {str(e)}"


@command(
    "execute_shell",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    CFG.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(CFG.workspace_directory):
        os.chdir(CFG.workspace_directory)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)
    return output


@command(
    "execute_shell_popen",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    CFG.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if CFG.workspace_directory not in current_dir:
        os.chdir(CFG.workspace_directory)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    do_not_show_output = subprocess.DEVNULL
    process = subprocess.Popen(
        command_line, shell=True, stdout=do_not_show_output, stderr=do_not_show_output
    )

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)

    return f"Subprocess started with PID:'{str(process.pid)}'"

def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")


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

@command("run_image", "Generate Image with stablediffusion", '"prompt": "<prompt>"')
def run_image(prompt: str, size: int = 768) -> str:
    """Generate an image with Stable Diffusion.

    Args:
        prompt (str): The prompt to use

    Returns:
        str: The filename of the image
    """
    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    workspace_directory = f"{visionhack}/auto_gpt_workspace"
    if str(workspace_directory) not in current_dir:
        os.chdir(workspace_directory)

    # Execute the stablediffusion.py script with the provided prompt
    command_line = f'python stablediffusion.py --prompt "{prompt}"'
    print(f"Executing command 'run_image' in working directory...")

    result = subprocess.run(command_line, capture_output=True, shell=True, encoding="utf-8")
    output = result.stdout

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

        return f"generated image saved to {filename}"
    else:
        return "Error: Generated image not found."
