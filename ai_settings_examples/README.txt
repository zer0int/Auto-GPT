- Copy to the root dir (/Auto-GPT), rename to ai_settings.yaml
- Recommended: Delete any generated images in Auto-GPT/images (if applicable) your stablediffusion/outputs local folder. 
  Prevents GPT-3.5 from tripping up due to initial generation being returned as 00005.png due to "leftovers" in the folder. GPT-4 won't care.

## A GENERAL NOTE ON IMAGES & CLIP
As you can see, some of the example images I provided in the /Auto-GPT/images folder contains excessive TEXT. CLIP loves text. CLIP is obsessed about reading text!
You can use this to steer CLIP, making sure CLIP 'sees' a sneaker indeed. Or you can make CLIP lose it and go nuts if you write "banana" on your sneaker.
STRG+F for "typographic attacks" in OpenAI's blog post at https://openai.com/research/multimodal-neurons to learn more about CLIP's AI-weirdness.



ai_settings_visual_websearch.yaml

Looks at the specified image from /Auto-GPT/images and gets a CLIP opinion about it. Will then make sense of the CLIP opinion, do websearch for similar images,
and finally attempt to download images. Doesn't actually work because google goes "shoo shoo, copyright, go away" - but you can still check out the URL / images in your browser.
A quick proof-of-concept to test (and probably be surprised by) the accuracy in which GPT makes sense of / understands CLIP's weird tokens (depending on image).


ai_settings_image_to_shape3D.yaml

Looks at /Auto-GPT/images 00001.png thru 00010.png one after the other, and makes a 3D .ply mesh based on the CLIP opinion about each image (saves .ply to /AutoGPT/images)
- The .png could be the images you created with stable diffusion during the last run, for example. For other options, check comments in /Auto-GPT/auto_gpt_workspace/SHAPErun.py

Add something along the lines of "Continue with run_clip on the generated image, and so on, as before." to the last goal to instruct Auto-GPT to run the task in a loop indefinitely.
Works perfectly well with GPT-4; may or may not randomly confuse GPT-3.5 ("sometimes it works, and the next time it doesn't" -> GPT-3.5 model limitation, most likely).


ai_settings_stablediffusion_local.yaml

Looks at /Auto-GPT/images 0001.png. Telling it to choose a random one is really pointless, but I left the info about more files in the goal prompt "just in case'.
Uses the CLIP opinion about image 0001.png, then generates an image with the local stable diffusion (replace run_image with generate_image for other options).
Takes the freshly generated image and gets a CLIP opinion about it -> prompt a new image -> get a CLIP opinion -> etc. 
(GPT-4: I stopped after $10 / 42 flawless iterations; GPT-3.5 has severe model limitations, but often manages 6-10 itt before doing something weird because it forgot what it was doing)

Add something along the lines of "Continue with run_clip on the generated image, and so on, as before." to the last goal to instruct Auto-GPT to run the task in a loop indefinitely.
Works perfectly well with GPT-4; may or may not randomly confuse GPT-3.5 ("sometimes it works, and the next time it doesn't" -> GPT-3.5 model limitation, most likely).

### NOTE: Your actual stable diffusion "output" folder will be trashed up with images, as I shutil.copyfile them instead of shutil.move.
### Why? I couldn't figure out how to implement a check that makes sure existing files are NOT overwritten in the Auto-GPT folder, but limit that to stable diffusion output images. 
### You're gonna have to manually clean out your redundant images in stablediffusion/outputs folder, for the time being. Sorry about that.


DEPRECATED-ai_settings-SD-local.yaml

Used to work with Auto-GPT v0.2.2 - I didn't get it to work with Auto-GPT 0.3.1. This will prompt to generate an image using the "official" generate_image command in Auto-GPT.
Otherwise, does the same as ai_settings_stablediffusion_local.yaml -- just using another method. The code is there in image_gen.py - if you wanna take a shot, feel free!
Add this line to your .env in ### IMAGE GENERATION PROVIDER ###:
IMAGE_PROVIDER=stablediffusion


ai_settings_MADCAP-AI-SOUP.yaml

### !!! EDIT auto_gpt_workspace/SHAPErun.py -- comment out the simple "return" statement, uncomment return statement that will return the filename to the AI. !!!

Watch GPT-4 masterfully juggle a mix of CLIP, stable diffusion, and Shap-E, generating images and then 3D meshes bashed on the previous image, then an image inspired by the 3D mesh, and so on.
Has multiple goals crammed into one goal prompt / more than 5 goals. Inevitably spams the AI with lots of information regarding an abundance of files generated during progress.
Can be used to demonstrate model limitation when used with GPT-3.5 (albeit it often works for a few iterations) vs. GPT-4 superiority in infinitely juggling a multitude of AI.




####################################
#####        WARNING           #####
####################################

While you probably shouldn't run Auto-GPT in "autonomous" mode, anyway, you'll probably also want to ACTUALLY proof-read the GPT-generated prompt before just approving it!
That is especially the case if you are not running local, and spamming offensive words might just get you banned from a text-to-image API.

!!! CLIP IS UNCENSORED. CLIP SEES WHATEVER CLIP WANTS TO SEE (doesn't have to be related to what *you* see), including anything you can imagine there to be seen as of the training dataset.

So a harmless image (your opinion) might lead to offensive, racist, biased, sexist output (CLIP opinion). Especially true if non-English text is present in the image.
More info on typographic attacks and why CLIP is so obsessed with text: https://openai.com/research/multimodal-neurons
Check the model-card.md and heed the warnings from OpenAI: https://github.com/openai/CLIP/blob/main/model-card.md

Run from CMD: C:\Users\You> python CLIP.py --image_path "path/to/Auto-GPT/images/pepe.png" for an example that "oh yes, CLIP knows - CLIP was trained on the internet".

PS: And yes, GPT-3.5 / GPT-4 will accept these terms and make a prompt with them. They might conclude "the CLIP opinion is not very useful" and try to do something else;
however, the AI can be persuaded to "use the CLIP tokens to make a prompt for run_image" via user feedback, and will just refrain from using blatant words like "rape". 
However, CLIP opinion often includes chained word tokens, like "instarape", which GPT accepts; and that will be understood by the CLIP inside stable diffusion et al. just as well. 
And likely by an API filter, too.

You have been warned. Do whatever floats your boat, but keep it limited to *your* boat - and don't blame me for getting kick-banned from any text-to-image API. That's all.