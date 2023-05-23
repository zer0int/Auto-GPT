import os
import sys
import argparse
import subprocess
sys.path.append(".././")
from autogpt.visionconfig import visionhack, visionhome

def run_gradcam(image_path, txt_path):
    # Define the command to run the script
    script_path = f"{visionhack}/autogpt/commands/CLIP_gradcam.py"
    command = ["python", script_path, "--image", image_path, "--txt", txt_path]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("The script ran successfully.")
    except subprocess.CalledProcessError:
        print("There was an error running the script.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CLIP_gradcam.py")
    parser.add_argument("--image", type=str, required=True, default="0001.png", help="The image to visualize heat map.")
    parser.add_argument("--txt", type=str, required=True, default="tokens_0001.txt", help="The tokens .txt to use to guide CLIP.")
    args = parser.parse_args()

    # Call function to run the script
    run_gradcam(args.image, args.txt)

if __name__ == "__main__":
    main()
