import os
import sys
import argparse
import subprocess
sys.path.append(".././")
from autogpt.visionconfig import visionhack, visionhome

# Parse the command line argument for the image path
arg_parser = argparse.ArgumentParser(description="Run CLIP script with subprocess")
arg_parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
args = arg_parser.parse_args()

clip_command = ["python", f"{visionhome}/CLIP.py", "--image_path", args.image_path]

result = subprocess.run(clip_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

if result.returncode == 0:
    output_filename = f"tokens_{os.path.splitext(os.path.basename(args.image_path))[0]}.txt"
    print(f"CLIP tokens saved to ./auto_gpt_workspace/{output_filename}.")
    #print(f"CLIP tokens saved to {visionhack}/auto_gpt_workspace/{output_filename}.")
    # Continue with processing the output tokens
else:
    print("CLIP script encountered an error.")
    print("Error details:", result.stderr)