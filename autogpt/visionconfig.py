### visionconfig for CLIP vision hack by zer0int

# The absolute path to the root directory of your Auto-GPT project

visionhack = "C:/Users/zer0int/Auto-GPT"


# The absolute path to your user home directory, i.e. "C:/Users/Anton"
# Copy CLIP.py and SHAPE.py (originally in /auto_gpt_workspace) to this user home directory!
# Required prequisites to install:
# https://github.com/openai/CLIP
# Optional:
# https://github.com/openai/shap-e
# https://github.com/Stability-AI/stablediffusion

visionhome = "C:/Users/zer0int"


# Your local stable diffusion, enter the absolute path to the root directory below
# in your .env ### IMAGE GENERATION PROVIDER ###, add this (without a leading #) to enable:
# IMAGE_PROVIDER=stablediffusion
# Will use the "old" v2-1_768-ema-pruned.ckpt & v2-inference-v.yaml by default - adjust at will in /auto_gpt_workspace/stablediffusion.py

stablehome = "C:/Users/zer0int/stablediffusionAutoGPT"


# OpenAI Shap-E text-to-3D: Just install per instructions in the repo, done.
# First run ever will download & build the Shap-E model inside /auto_gpt_workspace -- BE PATIENT!!