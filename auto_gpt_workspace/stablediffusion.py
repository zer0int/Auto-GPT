import os
import sys
import argparse
import subprocess
import re
import shutil
os.chdir(".././")
sys.path.append("./")
from autogpt.visionconfig import visionhack, stablehome
os.chdir(f"{stablehome}")

# Parse the command line argument for the prompt
arg_parser = argparse.ArgumentParser(description="Run stable diffusion script with subprocess")
arg_parser.add_argument("--prompt", type=str, required=True, help="Prompt for stable diffusion")
args = arg_parser.parse_args()

# I only pass prompt from GPT (see above); adjust & add other parameters below "secretly" without the AI knowing (and getting confused about it)!
stable_diffusion_command = [
    "python",
    "./scripts/txt2img.py",
    "--prompt", args.prompt,
    "--ckpt", "./models/v2-1_768-ema-pruned.ckpt",
    "--config", "./configs/stable-diffusion/v2-inference-v.yaml",
    "--n_samples", "1",
    "--n_iter", "1",
    "--H", "768",
    "--W", "768"
]

result = subprocess.run(stable_diffusion_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if result.returncode == 0:
    print("Stable diffusion image generation script completed successfully.")
    # Continue with processing the output image
else:
    print("Stable diffusion script encountered an error.")
    print("Error details:", result.stderr)
