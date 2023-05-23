import os
import sys
import argparse
import subprocess
sys.path.append(".././")
from autogpt.visionconfig import visionhack, visionhome

# Parse the command line argument for the prompt
arg_parser = argparse.ArgumentParser(description="Run SHAPE script with subprocess")
arg_parser.add_argument("--prompt", type=str, required=True, help="The prompt for Shap-E")
args = arg_parser.parse_args()

shape_command = ["python", f"{visionhome}/SHAPE.py", "--prompt", args.prompt]

result = subprocess.run(shape_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

if result.returncode == 0:
    output_filename = result.stdout.strip()

    print(f"SHAPE image generated successfully.")

    # You can use [below] to return one view (image) of the 3D model the AI created
    # Can be used in conjuction with goal: "Get a CLIP opinion about the image you created with shape | make prompt | run_shape to make a new image based on the CLIP opinion"
    # The AI doesn't need to know about the fact it is making 3D models! Also, [next line] will likely confuse GPT-3.5 -- I'd recommend: Use with GPT-4 only.

    #print(f"SHAPE image saved to {output_filename}.")
    #print(f"SHAPE image saved to {visionhack}/images/{output_filename}.")

else:
    print("SHAPE script encountered an error.")
    print("Error details:", result.stderr)
