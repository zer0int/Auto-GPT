# Auto-GPT with CLIP vision for GPT-4 and GPT-3.5
![BANNER-CLIP-GPT](https://github.com/zer0int/Auto-GPT/assets/132047210/63e224e7-deba-42b0-a33a-84d07e5efbd2)
------
## A pseudo-multimodal Auto-GPT 
(We're essentially just prompting GPT with CLIP's "opinion" tokens for what it "sees" in an image; which is surprisingly effective, nevertheless!)

- GPT autonomous infinite variation prompting of text-to-image and text-to-3D 
- GPT visual websearch based on image content 
- your_idea_here

Please tag me on [Twitter: @zer0int1](https://twitter.com/zer0int1) if you use this for anything; I'd love to see it!

Demo videos:
---------------
[<img src="http://img.youtube.com/vi/fzFuTe1E5tU/0.jpg" width="240" height="180">](http://www.youtube.com/watch?v=fzFuTe1E5tU "Infinite shoe design with CLIP + Stable Diffusion") 🔛 [<img src="http://img.youtube.com/vi/0C5WdhPfvIk/0.jpg" width="240" height="180">](http://www.youtube.com/watch?v=0C5WdhPfvIk "Infinite 3D mesh shoe design with CLIP + Shap-E")
----------------

## Steps to install / use

### Prerequisite:

- ✅ [CLIP by OpenAI](https://github.com/openai/CLIP)

### Optional Prerequisites:
Depending on your use case, you might also need:

- ✔️ [Shap-E by OpenAI](https://github.com/openai/shap-e) (Text-to-3D)
- ✔️ [Stable Diffusion by Stability-AI](https://github.com/Stability-AI/stablediffusion) (Text-to-Image, for commandline & running local as implemented in this repo)

Other third-party credits: 
- CLIP Gradient Ascent: Adaptation of the original notebook "Closed Test Ascending CLIPtext" by [@advadnoun](https://twitter.com/advadnoun)
- CLIP GradCAM: [https://github.com/kevinzakka/clip_playground](https://github.com/kevinzakka/clip_playground)

### 🤔❗ Usage:

0. Ensure prequisites (above repos) are installed and working
1. Put (git-clone, download-zip-extract) this in your C:/User/JohnDoe or equivalent "user home" folder ("z" is no longer a null byte, I removed it, so your username can be with a "z", like mine)
2. Edit Auto-GPT/auto_gpt_workspace/CLIP.py according to your hardware (VRAM requirement table inside; yes you can run a small CLIP with 6 GB of VRAM!)
3. From Auto-GPT/auto_gpt_workspace, copy CLIP.py and (if applicable) SHAPE.py to your C:/User/JohnDoe (one level above the Auto-GPT folder)
4. Edit Auto-GPT/autogpt/visionconfig.py (Should be straightforward and self-explanatory, define the absolute paths on your local system in this config)
5. In your .env, remove the comment and set EXECUTE_LOCAL_COMMANDS=True and RESTRICT_TO_WORKSPACE=False
6. Pick one of the .yaml files from Auto-GPT/ai_settings_examples (see README.TXT for details!), copy it to the main Auto-GPT folder and rename it, so you have e.g. C:/User/JohnDoe/Auto-GPT/ai_settings.yaml
7. (Optional) put your own images (filenames as per the ai_settings.yaml) in the Auto-GPT/images folder (or use the example images I provided)
8. (Optional, case: running local stable diffusion): Edit Auto-GPT/auto_gpt_workspace/stablediffusion.py to match the model / config of SD you want to use
8. (Optional, recommended) Make sure that everything is working by running the scripts independently outside of Auto-GPT (see below)

### Verify everything works via cmd -> cd into Auto-GPT/auto_gpt_workspace and run:

```bash
python CLIPrun.py --image_path "C:/Users/JohnDoe/Auto-GPT/images/0001.png"
```
Replace JohnDoe with YourUserName, wait a minute or two (depending on GPU and settings you made in CLIP.py)
-> You should now have a CLIP opinion as tokens_0001.txt in Auto-GPT/auto_gpt_workspace

Optional:

```bash
python SHAPErun.py --prompt "A Pontiac Firebird car"
```
You should find s_000.png and the respective .ply in the Auto-GPT/images folder.
When you run this FOR THE FIRST TIME EVER, it will download & build the Shap-E models in Auto-GPT/auto_gpt_workspace/shap_e_model_cache
--> Please be patient (minutes, depending on your internet speed)!!

-----------------
![0001-RN50x4_L1-exploemails](https://github.com/zer0int/Auto-GPT/assets/132047210/7290c63a-aa86-4913-beee-435a68fe7c2f) ![0001-RN50x4_L4-sett](https://github.com/zer0int/Auto-GPT/assets/132047210/54373152-8c3b-40c4-a757-938c5a29963d)

🤖👀 BONUS: See what CLIP sees by computing (fast!) a heatmap highlighting which regions in the image activate the most to a given caption.
```bash
python manual_gradcam.py --image "0001.png" --txt "tokens_0001.txt"
```
Caption = tokens CLIP 'saw' in the image (returned "opinion" tokens_XXXXX.txt of GPT using "run_clip" on XXXXX.png in Auto-GPT)
If you're wondering WTF CLIP saw in your image, and where - run this in a seperate command prompt "on the side" and according to what GPT last used in Auto-GPT.
Will dump heatmap images for all CLIP tokens of all four saliency layers of the CLIP model in the Auto-GPT/GradCAM folder.
For GradCAM requirements, see Auto-GPT/autogpt/commands/CLIP_gradcam.py -- adaptation of an ipynb notebook, pip install requirements left in as comments at the very top

ToDo: Implement as "y -D" option that Auto-GPT accepts, same as "y -N", to execute after the next time run_clip is executed.

-----
## ❓ Important tips and troubleshooting, including model limitations

- Oddly enough, the relative output .\auto_gpt_workspace\clip_tokens.txt will ensure GPT-3.5 does not get confused and not knowing where CLIP token "opinion" is. 
  GPT-4, however, will once try to read_file from the wrong place in the beginning. Simply approve with y, AI will "think" about file-not-found, correct itself, and never make that mistake again.
  Sorry about a small waste of GPT-4 tokens - but this is the best way to make sure it works out of the box for both GPT models.

- Make sure the folder structure is exactly as mentioned above. It's a delicate thing with executing subprocesses (.py files) from different locations.

- Delete the auto-gpt.json in Auto-GPT/auto_gpt_workspace if you change the .yaml or encounter issues.

- For local stable diffusion: I am using shutil.copyfile instead of shutil.move, meaning, I am trashing up your stablediffusion/outputs folder as I copy the images to Auto-GPT/images.
  Why? Because I couldn't figure out a way that will ONLY check for existing stable diffusion images, e.g. 00001.png but NOT 0001.png. Naming the files slightly more complex, like SD_00001.png, instead confuses GPT-3.5.
  So: Better have a bit of redundancy trash than files overwritten, right? Feel free to implement something that works, if you know how - I'd be delighted!

----


## CLIP IS UNCENSORED
⚠️💣💥⚠️
- CLIP SEES WHATEVER CLIP WANTS TO SEE (doesn't have to be related to what *you* see) 🤯

⚠️⚠️ WARNING ABOUT "BIAS" AND "HARMFUL" OUTPUT IN PRE-TRAINED, UNCENSORED CLIP MODELS.

While you probably shouldn't run Auto-GPT in "autonomous" mode, anyway, you'll probably also want to ACTUALLY proof-read the GPT-generated prompt ❗❗ carefully rather than just approving it!
That is especially the case if you are not running local, and spamming offensive words might just get you banned from a text-to-image API ❗❗

So a harmless image (your opinion) might lead to offensive, racist, biased, sexist output (CLIP opinion) ❗. Especially true if non-English text is present in the image.
- 👉 More info on typographic attacks and why CLIP is so obsessed with text: [Multimodal Neurons](https://openai.com/research/multimodal-neurons)
- 👉 Check the model-card.md and heed the warnings from OpenAI: [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md)

Use the above CLIPrun.py with pepe.png for an example that shouldn't be too toxic, but proves a point with regard to "oh yes, CLIP knows - CLIP was trained on the internet".

PS: And yes, GPT-3.5 / GPT-4 will accept these terms and make a prompt with them. They might conclude "the CLIP opinion is not very useful" and try to do something else;
however, the AI can be persuaded to "use the CLIP tokens to make a prompt for run_image" via user feedback, and will then only refrain from using blatantly offensive words like "r*pe". 
However, CLIP opinion often includes chained "longword" tokens, like e.g. "instarape" - which GPT accepts, and that will in turn be understood by the CLIP inside stable diffusion et al just as well. 
...And likely by an API filter, too.

You have been warned. Do whatever floats your boat, but keep it limited to *your* boat - and don't blame me for getting kick-banned from any text-to-image API. That's all. ❗
---
⚠️⚠️⚠️⚠️⚠️⚠️
----------------
Original README.MD
---


# Auto-GPT: An Autonomous GPT-4 Experiment
[![Official Website](https://img.shields.io/badge/Official%20Website-agpt.co-blue?style=flat&logo=world&logoColor=white)](https://agpt.co)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/Significant-Gravitas/Auto-GPT/ci.yml?label=unit%20tests)](https://github.com/Significant-Gravitas/Auto-GPT/actions/workflows/ci.yml)
[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![GitHub Repo stars](https://img.shields.io/github/stars/Significant-Gravitas/auto-gpt?style=social)](https://github.com/Significant-Gravitas/Auto-GPT/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/siggravitas?style=social)](https://twitter.com/SigGravitas)

## 💡 Get help - [Q&A](https://github.com/Significant-Gravitas/Auto-GPT/discussions/categories/q-a) or [Discord 💬](https://discord.gg/autogpt)

<hr/>

### 🔴 🔴 🔴  Urgent: USE `stable` not `master`  🔴 🔴 🔴

**Download the latest `stable` release from here: https://github.com/Significant-Gravitas/Auto-GPT/releases/latest.**
The `master` branch may often be in a **broken** state.

<hr/>


Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

<h2 align="center"> Demo April 16th 2023 </h2>

https://user-images.githubusercontent.com/70048414/232352935-55c6bf7c-3958-406e-8610-0913475a0b05.mp4

Demo made by <a href=https://twitter.com/BlakeWerlinger>Blake Werlinger</a>

<h2 align="center"> 💖 Help Fund Auto-GPT's Development 💖</h2>
<p align="center">
If you can spare a coffee, you can help to cover the costs of developing Auto-GPT and help to push the boundaries of fully autonomous AI!
Your support is greatly appreciated. Development of this free, open-source project is made possible by all the <a href="https://github.com/Significant-Gravitas/Auto-GPT/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/Torantulino">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/Torantulino">click here</a>.
</p>


<p align="center">
<div align="center" class="logo-container">
<a href="https://www.zilliz.com/">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234158272-7917382e-ff80-469e-8d8c-94f4477b8b5a.png">
  <img src="https://user-images.githubusercontent.com/22963551/234158222-30e2d7a7-f0a9-433d-a305-e3aa0b194444.png" height="40px" alt="Zilliz" />
</picture>
</a>

<a href="https://roost.ai">
<img src="https://user-images.githubusercontent.com/22963551/234180283-b58cb03c-c95a-4196-93c1-28b52a388e9d.png" height="40px" alt="Roost.AI" />
</a>
<a href="https://nuclei.ai/">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234153428-24a6f31d-c0c6-4c9b-b3f4-9110148f67b4.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181283-691c5d71-ca94-4646-a1cf-6e818bd86faa.png" height="40px" alt="NucleiAI" />
</picture>
</a>

<a href="https://www.algohash.org/">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234180375-1365891c-0ba6-4d49-94c3-847c85fe03b0.png" >
  <img src="https://user-images.githubusercontent.com/22963551/234180359-143e4a7a-4a71-4830-99c8-9b165cde995f.png" height="40px" alt="Algohash" />
</picture>
</a>

<a href="https://www.typingmind.com/?utm_source=autogpt">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/233202971-61e77209-58a0-47d9-9f7e-dd081111437b.png">
  <img src="https://user-images.githubusercontent.com/22963551/234157731-f908b5db-8fe7-4036-89b6-7b2a21f87e3a.png" height="40px" alt="TypingMind" />
</picture>
</a>

<a href="https://github.com/weaviate/weaviate">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234181699-3d7f6ea8-5a7f-4e98-b812-37be1081be4b.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181695-fc895159-b921-4895-9a13-65e6eff5b0e7.png" height="40px" alt="TypingMind" />
</picture>
</a>

</div>
</br>



<p align="center"><a href="https://github.com/robinicus"><img src="https://avatars.githubusercontent.com/robinicus?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/0xmatchmaker"><img src="https://avatars.githubusercontent.com/0xmatchmaker?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/jazgarewal"><img src="https://avatars.githubusercontent.com/jazgarewal?v=4" width="50px" alt="jazgarewal" /></a>&nbsp;&nbsp;<a href="https://github.com/MayurVirkar"><img src="https://avatars.githubusercontent.com/MayurVirkar?v=4" width="50px" alt="MayurVirkar" /></a>&nbsp;&nbsp;<a href="https://github.com/avy-ai"><img src="https://avatars.githubusercontent.com/avy-ai?v=4" width="50px" alt="avy-ai" /></a>&nbsp;&nbsp;<a href="https://github.com/TheStoneMX"><img src="https://avatars.githubusercontent.com/TheStoneMX?v=4" width="50px" alt="TheStoneMX" /></a>&nbsp;&nbsp;<a href="https://github.com/goldenrecursion"><img src="https://avatars.githubusercontent.com/goldenrecursion?v=4" width="50px" alt="goldenrecursion" /></a>&nbsp;&nbsp;<a href="https://github.com/MatthewAgs"><img src="https://avatars.githubusercontent.com/MatthewAgs?v=4" width="50px" alt="MatthewAgs" /></a>&nbsp;&nbsp;<a href="https://github.com/eelbaz"><img src="https://avatars.githubusercontent.com/eelbaz?v=4" width="50px" alt="eelbaz" /></a>&nbsp;&nbsp;<a href="https://github.com/rapidstartup"><img src="https://avatars.githubusercontent.com/rapidstartup?v=4" width="50px" alt="rapidstartup" /></a>&nbsp;&nbsp;<a href="https://github.com/gklab"><img src="https://avatars.githubusercontent.com/gklab?v=4" width="50px" alt="gklab" /></a>&nbsp;&nbsp;<a href="https://github.com/VoiceBeer"><img src="https://avatars.githubusercontent.com/VoiceBeer?v=4" width="50px" alt="VoiceBeer" /></a>&nbsp;&nbsp;<a href="https://github.com/DailyBotHQ"><img src="https://avatars.githubusercontent.com/DailyBotHQ?v=4" width="50px" alt="DailyBotHQ" /></a>&nbsp;&nbsp;<a href="https://github.com/lucas-chu"><img src="https://avatars.githubusercontent.com/lucas-chu?v=4" width="50px" alt="lucas-chu" /></a>&nbsp;&nbsp;<a href="https://github.com/knifour"><img src="https://avatars.githubusercontent.com/knifour?v=4" width="50px" alt="knifour" /></a>&nbsp;&nbsp;<a href="https://github.com/refinery1"><img src="https://avatars.githubusercontent.com/refinery1?v=4" width="50px" alt="refinery1" /></a>&nbsp;&nbsp;<a href="https://github.com/st617"><img src="https://avatars.githubusercontent.com/st617?v=4" width="50px" alt="st617" /></a>&nbsp;&nbsp;<a href="https://github.com/neodenit"><img src="https://avatars.githubusercontent.com/neodenit?v=4" width="50px" alt="neodenit" /></a>&nbsp;&nbsp;<a href="https://github.com/CrazySwami"><img src="https://avatars.githubusercontent.com/CrazySwami?v=4" width="50px" alt="CrazySwami" /></a>&nbsp;&nbsp;<a href="https://github.com/Heitechsoft"><img src="https://avatars.githubusercontent.com/Heitechsoft?v=4" width="50px" alt="Heitechsoft" /></a>&nbsp;&nbsp;<a href="https://github.com/RealChrisSean"><img src="https://avatars.githubusercontent.com/RealChrisSean?v=4" width="50px" alt="RealChrisSean" /></a>&nbsp;&nbsp;<a href="https://github.com/abhinav-pandey29"><img src="https://avatars.githubusercontent.com/abhinav-pandey29?v=4" width="50px" alt="abhinav-pandey29" /></a>&nbsp;&nbsp;<a href="https://github.com/Explorergt92"><img src="https://avatars.githubusercontent.com/Explorergt92?v=4" width="50px" alt="Explorergt92" /></a>&nbsp;&nbsp;<a href="https://github.com/SparkplanAI"><img src="https://avatars.githubusercontent.com/SparkplanAI?v=4" width="50px" alt="SparkplanAI" /></a>&nbsp;&nbsp;<a href="https://github.com/crizzler"><img src="https://avatars.githubusercontent.com/crizzler?v=4" width="50px" alt="crizzler" /></a>&nbsp;&nbsp;<a href="https://github.com/kreativai"><img src="https://avatars.githubusercontent.com/kreativai?v=4" width="50px" alt="kreativai" /></a>&nbsp;&nbsp;<a href="https://github.com/omphos"><img src="https://avatars.githubusercontent.com/omphos?v=4" width="50px" alt="omphos" /></a>&nbsp;&nbsp;<a href="https://github.com/Jahmazon"><img src="https://avatars.githubusercontent.com/Jahmazon?v=4" width="50px" alt="Jahmazon" /></a>&nbsp;&nbsp;<a href="https://github.com/tjarmain"><img src="https://avatars.githubusercontent.com/tjarmain?v=4" width="50px" alt="tjarmain" /></a>&nbsp;&nbsp;<a href="https://github.com/ddtarazona"><img src="https://avatars.githubusercontent.com/ddtarazona?v=4" width="50px" alt="ddtarazona" /></a>&nbsp;&nbsp;<a href="https://github.com/saten-private"><img src="https://avatars.githubusercontent.com/saten-private?v=4" width="50px" alt="saten-private" /></a>&nbsp;&nbsp;<a href="https://github.com/anvarazizov"><img src="https://avatars.githubusercontent.com/anvarazizov?v=4" width="50px" alt="anvarazizov" /></a>&nbsp;&nbsp;<a href="https://github.com/lazzacapital"><img src="https://avatars.githubusercontent.com/lazzacapital?v=4" width="50px" alt="lazzacapital" /></a>&nbsp;&nbsp;<a href="https://github.com/m"><img src="https://avatars.githubusercontent.com/m?v=4" width="50px" alt="m" /></a>&nbsp;&nbsp;<a href="https://github.com/Pythagora-io"><img src="https://avatars.githubusercontent.com/Pythagora-io?v=4" width="50px" alt="Pythagora-io" /></a>&nbsp;&nbsp;<a href="https://github.com/Web3Capital"><img src="https://avatars.githubusercontent.com/Web3Capital?v=4" width="50px" alt="Web3Capital" /></a>&nbsp;&nbsp;<a href="https://github.com/toverly1"><img src="https://avatars.githubusercontent.com/toverly1?v=4" width="50px" alt="toverly1" /></a>&nbsp;&nbsp;<a href="https://github.com/digisomni"><img src="https://avatars.githubusercontent.com/digisomni?v=4" width="50px" alt="digisomni" /></a>&nbsp;&nbsp;<a href="https://github.com/concreit"><img src="https://avatars.githubusercontent.com/concreit?v=4" width="50px" alt="concreit" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRobidas"><img src="https://avatars.githubusercontent.com/LeeRobidas?v=4" width="50px" alt="LeeRobidas" /></a>&nbsp;&nbsp;<a href="https://github.com/Josecodesalot"><img src="https://avatars.githubusercontent.com/Josecodesalot?v=4" width="50px" alt="Josecodesalot" /></a>&nbsp;&nbsp;<a href="https://github.com/dexterityx"><img src="https://avatars.githubusercontent.com/dexterityx?v=4" width="50px" alt="dexterityx" /></a>&nbsp;&nbsp;<a href="https://github.com/rickscode"><img src="https://avatars.githubusercontent.com/rickscode?v=4" width="50px" alt="rickscode" /></a>&nbsp;&nbsp;<a href="https://github.com/Brodie0"><img src="https://avatars.githubusercontent.com/Brodie0?v=4" width="50px" alt="Brodie0" /></a>&nbsp;&nbsp;<a href="https://github.com/FSTatSBS"><img src="https://avatars.githubusercontent.com/FSTatSBS?v=4" width="50px" alt="FSTatSBS" /></a>&nbsp;&nbsp;<a href="https://github.com/nocodeclarity"><img src="https://avatars.githubusercontent.com/nocodeclarity?v=4" width="50px" alt="nocodeclarity" /></a>&nbsp;&nbsp;<a href="https://github.com/jsolejr"><img src="https://avatars.githubusercontent.com/jsolejr?v=4" width="50px" alt="jsolejr" /></a>&nbsp;&nbsp;<a href="https://github.com/amr-elsehemy"><img src="https://avatars.githubusercontent.com/amr-elsehemy?v=4" width="50px" alt="amr-elsehemy" /></a>&nbsp;&nbsp;<a href="https://github.com/RawBanana"><img src="https://avatars.githubusercontent.com/RawBanana?v=4" width="50px" alt="RawBanana" /></a>&nbsp;&nbsp;<a href="https://github.com/horazius"><img src="https://avatars.githubusercontent.com/horazius?v=4" width="50px" alt="horazius" /></a>&nbsp;&nbsp;<a href="https://github.com/SwftCoins"><img src="https://avatars.githubusercontent.com/SwftCoins?v=4" width="50px" alt="SwftCoins" /></a>&nbsp;&nbsp;<a href="https://github.com/tob-le-rone"><img src="https://avatars.githubusercontent.com/tob-le-rone?v=4" width="50px" alt="tob-le-rone" /></a>&nbsp;&nbsp;<a href="https://github.com/RThaweewat"><img src="https://avatars.githubusercontent.com/RThaweewat?v=4" width="50px" alt="RThaweewat" /></a>&nbsp;&nbsp;<a href="https://github.com/jun784"><img src="https://avatars.githubusercontent.com/jun784?v=4" width="50px" alt="jun784" /></a>&nbsp;&nbsp;<a href="https://github.com/joaomdmoura"><img src="https://avatars.githubusercontent.com/joaomdmoura?v=4" width="50px" alt="joaomdmoura" /></a>&nbsp;&nbsp;<a href="https://github.com/rejunity"><img src="https://avatars.githubusercontent.com/rejunity?v=4" width="50px" alt="rejunity" /></a>&nbsp;&nbsp;<a href="https://github.com/mathewhawkins"><img src="https://avatars.githubusercontent.com/mathewhawkins?v=4" width="50px" alt="mathewhawkins" /></a>&nbsp;&nbsp;<a href="https://github.com/caitlynmeeks"><img src="https://avatars.githubusercontent.com/caitlynmeeks?v=4" width="50px" alt="caitlynmeeks" /></a>&nbsp;&nbsp;<a href="https://github.com/jd3655"><img src="https://avatars.githubusercontent.com/jd3655?v=4" width="50px" alt="jd3655" /></a>&nbsp;&nbsp;<a href="https://github.com/Odin519Tomas"><img src="https://avatars.githubusercontent.com/Odin519Tomas?v=4" width="50px" alt="Odin519Tomas" /></a>&nbsp;&nbsp;<a href="https://github.com/DataMetis"><img src="https://avatars.githubusercontent.com/DataMetis?v=4" width="50px" alt="DataMetis" /></a>&nbsp;&nbsp;<a href="https://github.com/webbcolton"><img src="https://avatars.githubusercontent.com/webbcolton?v=4" width="50px" alt="webbcolton" /></a>&nbsp;&nbsp;<a href="https://github.com/rocks6"><img src="https://avatars.githubusercontent.com/rocks6?v=4" width="50px" alt="rocks6" /></a>&nbsp;&nbsp;<a href="https://github.com/cxs"><img src="https://avatars.githubusercontent.com/cxs?v=4" width="50px" alt="cxs" /></a>&nbsp;&nbsp;<a href="https://github.com/fruition"><img src="https://avatars.githubusercontent.com/fruition?v=4" width="50px" alt="fruition" /></a>&nbsp;&nbsp;<a href="https://github.com/nnkostov"><img src="https://avatars.githubusercontent.com/nnkostov?v=4" width="50px" alt="nnkostov" /></a>&nbsp;&nbsp;<a href="https://github.com/morcos"><img src="https://avatars.githubusercontent.com/morcos?v=4" width="50px" alt="morcos" /></a>&nbsp;&nbsp;<a href="https://github.com/pingbotan"><img src="https://avatars.githubusercontent.com/pingbotan?v=4" width="50px" alt="pingbotan" /></a>&nbsp;&nbsp;<a href="https://github.com/maxxflyer"><img src="https://avatars.githubusercontent.com/maxxflyer?v=4" width="50px" alt="maxxflyer" /></a>&nbsp;&nbsp;<a href="https://github.com/tommi-joentakanen"><img src="https://avatars.githubusercontent.com/tommi-joentakanen?v=4" width="50px" alt="tommi-joentakanen" /></a>&nbsp;&nbsp;<a href="https://github.com/hunteraraujo"><img src="https://avatars.githubusercontent.com/hunteraraujo?v=4" width="50px" alt="hunteraraujo" /></a>&nbsp;&nbsp;<a href="https://github.com/projectonegames"><img src="https://avatars.githubusercontent.com/projectonegames?v=4" width="50px" alt="projectonegames" /></a>&nbsp;&nbsp;<a href="https://github.com/tullytim"><img src="https://avatars.githubusercontent.com/tullytim?v=4" width="50px" alt="tullytim" /></a>&nbsp;&nbsp;<a href="https://github.com/comet-ml"><img src="https://avatars.githubusercontent.com/comet-ml?v=4" width="50px" alt="comet-ml" /></a>&nbsp;&nbsp;<a href="https://github.com/thepok"><img src="https://avatars.githubusercontent.com/thepok?v=4" width="50px" alt="thepok" /></a>&nbsp;&nbsp;<a href="https://github.com/prompthero"><img src="https://avatars.githubusercontent.com/prompthero?v=4" width="50px" alt="prompthero" /></a>&nbsp;&nbsp;<a href="https://github.com/sunchongren"><img src="https://avatars.githubusercontent.com/sunchongren?v=4" width="50px" alt="sunchongren" /></a>&nbsp;&nbsp;<a href="https://github.com/neverinstall"><img src="https://avatars.githubusercontent.com/neverinstall?v=4" width="50px" alt="neverinstall" /></a>&nbsp;&nbsp;<a href="https://github.com/josephcmiller2"><img src="https://avatars.githubusercontent.com/josephcmiller2?v=4" width="50px" alt="josephcmiller2" /></a>&nbsp;&nbsp;<a href="https://github.com/yx3110"><img src="https://avatars.githubusercontent.com/yx3110?v=4" width="50px" alt="yx3110" /></a>&nbsp;&nbsp;<a href="https://github.com/MBassi91"><img src="https://avatars.githubusercontent.com/MBassi91?v=4" width="50px" alt="MBassi91" /></a>&nbsp;&nbsp;<a href="https://github.com/SpacingLily"><img src="https://avatars.githubusercontent.com/SpacingLily?v=4" width="50px" alt="SpacingLily" /></a>&nbsp;&nbsp;<a href="https://github.com/arthur-x88"><img src="https://avatars.githubusercontent.com/arthur-x88?v=4" width="50px" alt="arthur-x88" /></a>&nbsp;&nbsp;<a href="https://github.com/ciscodebs"><img src="https://avatars.githubusercontent.com/ciscodebs?v=4" width="50px" alt="ciscodebs" /></a>&nbsp;&nbsp;<a href="https://github.com/christian-gheorghe"><img src="https://avatars.githubusercontent.com/christian-gheorghe?v=4" width="50px" alt="christian-gheorghe" /></a>&nbsp;&nbsp;<a href="https://github.com/EngageStrategies"><img src="https://avatars.githubusercontent.com/EngageStrategies?v=4" width="50px" alt="EngageStrategies" /></a>&nbsp;&nbsp;<a href="https://github.com/jondwillis"><img src="https://avatars.githubusercontent.com/jondwillis?v=4" width="50px" alt="jondwillis" /></a>&nbsp;&nbsp;<a href="https://github.com/Cameron-Fulton"><img src="https://avatars.githubusercontent.com/Cameron-Fulton?v=4" width="50px" alt="Cameron-Fulton" /></a>&nbsp;&nbsp;<a href="https://github.com/AryaXAI"><img src="https://avatars.githubusercontent.com/AryaXAI?v=4" width="50px" alt="AryaXAI" /></a>&nbsp;&nbsp;<a href="https://github.com/AuroraHolding"><img src="https://avatars.githubusercontent.com/AuroraHolding?v=4" width="50px" alt="AuroraHolding" /></a>&nbsp;&nbsp;<a href="https://github.com/Mr-Bishop42"><img src="https://avatars.githubusercontent.com/Mr-Bishop42?v=4" width="50px" alt="Mr-Bishop42" /></a>&nbsp;&nbsp;<a href="https://github.com/doverhq"><img src="https://avatars.githubusercontent.com/doverhq?v=4" width="50px" alt="doverhq" /></a>&nbsp;&nbsp;<a href="https://github.com/johnculkin"><img src="https://avatars.githubusercontent.com/johnculkin?v=4" width="50px" alt="johnculkin" /></a>&nbsp;&nbsp;<a href="https://github.com/marv-technology"><img src="https://avatars.githubusercontent.com/marv-technology?v=4" width="50px" alt="marv-technology" /></a>&nbsp;&nbsp;<a href="https://github.com/ikarosai"><img src="https://avatars.githubusercontent.com/ikarosai?v=4" width="50px" alt="ikarosai" /></a>&nbsp;&nbsp;<a href="https://github.com/ColinConwell"><img src="https://avatars.githubusercontent.com/ColinConwell?v=4" width="50px" alt="ColinConwell" /></a>&nbsp;&nbsp;<a href="https://github.com/humungasaurus"><img src="https://avatars.githubusercontent.com/humungasaurus?v=4" width="50px" alt="humungasaurus" /></a>&nbsp;&nbsp;<a href="https://github.com/terpsfreak"><img src="https://avatars.githubusercontent.com/terpsfreak?v=4" width="50px" alt="terpsfreak" /></a>&nbsp;&nbsp;<a href="https://github.com/iddelacruz"><img src="https://avatars.githubusercontent.com/iddelacruz?v=4" width="50px" alt="iddelacruz" /></a>&nbsp;&nbsp;<a href="https://github.com/thisisjeffchen"><img src="https://avatars.githubusercontent.com/thisisjeffchen?v=4" width="50px" alt="thisisjeffchen" /></a>&nbsp;&nbsp;<a href="https://github.com/nicoguyon"><img src="https://avatars.githubusercontent.com/nicoguyon?v=4" width="50px" alt="nicoguyon" /></a>&nbsp;&nbsp;<a href="https://github.com/arjunb023"><img src="https://avatars.githubusercontent.com/arjunb023?v=4" width="50px" alt="arjunb023" /></a>&nbsp;&nbsp;<a href="https://github.com/Nalhos"><img src="https://avatars.githubusercontent.com/Nalhos?v=4" width="50px" alt="Nalhos" /></a>&nbsp;&nbsp;<a href="https://github.com/belharethsami"><img src="https://avatars.githubusercontent.com/belharethsami?v=4" width="50px" alt="belharethsami" /></a>&nbsp;&nbsp;<a href="https://github.com/Mobivs"><img src="https://avatars.githubusercontent.com/Mobivs?v=4" width="50px" alt="Mobivs" /></a>&nbsp;&nbsp;<a href="https://github.com/txtr99"><img src="https://avatars.githubusercontent.com/txtr99?v=4" width="50px" alt="txtr99" /></a>&nbsp;&nbsp;<a href="https://github.com/ntwrite"><img src="https://avatars.githubusercontent.com/ntwrite?v=4" width="50px" alt="ntwrite" /></a>&nbsp;&nbsp;<a href="https://github.com/founderblocks-sils"><img src="https://avatars.githubusercontent.com/founderblocks-sils?v=4" width="50px" alt="founderblocks-sils" /></a>&nbsp;&nbsp;<a href="https://github.com/kMag410"><img src="https://avatars.githubusercontent.com/kMag410?v=4" width="50px" alt="kMag410" /></a>&nbsp;&nbsp;<a href="https://github.com/angiaou"><img src="https://avatars.githubusercontent.com/angiaou?v=4" width="50px" alt="angiaou" /></a>&nbsp;&nbsp;<a href="https://github.com/garythebat"><img src="https://avatars.githubusercontent.com/garythebat?v=4" width="50px" alt="garythebat" /></a>&nbsp;&nbsp;<a href="https://github.com/lmaugustin"><img src="https://avatars.githubusercontent.com/lmaugustin?v=4" width="50px" alt="lmaugustin" /></a>&nbsp;&nbsp;<a href="https://github.com/shawnharmsen"><img src="https://avatars.githubusercontent.com/shawnharmsen?v=4" width="50px" alt="shawnharmsen" /></a>&nbsp;&nbsp;<a href="https://github.com/clortegah"><img src="https://avatars.githubusercontent.com/clortegah?v=4" width="50px" alt="clortegah" /></a>&nbsp;&nbsp;<a href="https://github.com/MetaPath01"><img src="https://avatars.githubusercontent.com/MetaPath01?v=4" width="50px" alt="MetaPath01" /></a>&nbsp;&nbsp;<a href="https://github.com/sekomike910"><img src="https://avatars.githubusercontent.com/sekomike910?v=4" width="50px" alt="sekomike910" /></a>&nbsp;&nbsp;<a href="https://github.com/MediConCenHK"><img src="https://avatars.githubusercontent.com/MediConCenHK?v=4" width="50px" alt="MediConCenHK" /></a>&nbsp;&nbsp;<a href="https://github.com/svpermari0"><img src="https://avatars.githubusercontent.com/svpermari0?v=4" width="50px" alt="svpermari0" /></a>&nbsp;&nbsp;<a href="https://github.com/jacobyoby"><img src="https://avatars.githubusercontent.com/jacobyoby?v=4" width="50px" alt="jacobyoby" /></a>&nbsp;&nbsp;<a href="https://github.com/turintech"><img src="https://avatars.githubusercontent.com/turintech?v=4" width="50px" alt="turintech" /></a>&nbsp;&nbsp;<a href="https://github.com/allenstecat"><img src="https://avatars.githubusercontent.com/allenstecat?v=4" width="50px" alt="allenstecat" /></a>&nbsp;&nbsp;<a href="https://github.com/CatsMeow492"><img src="https://avatars.githubusercontent.com/CatsMeow492?v=4" width="50px" alt="CatsMeow492" /></a>&nbsp;&nbsp;<a href="https://github.com/tommygeee"><img src="https://avatars.githubusercontent.com/tommygeee?v=4" width="50px" alt="tommygeee" /></a>&nbsp;&nbsp;<a href="https://github.com/judegomila"><img src="https://avatars.githubusercontent.com/judegomila?v=4" width="50px" alt="judegomila" /></a>&nbsp;&nbsp;<a href="https://github.com/cfarquhar"><img src="https://avatars.githubusercontent.com/cfarquhar?v=4" width="50px" alt="cfarquhar" /></a>&nbsp;&nbsp;<a href="https://github.com/ZoneSixGames"><img src="https://avatars.githubusercontent.com/ZoneSixGames?v=4" width="50px" alt="ZoneSixGames" /></a>&nbsp;&nbsp;<a href="https://github.com/kenndanielso"><img src="https://avatars.githubusercontent.com/kenndanielso?v=4" width="50px" alt="kenndanielso" /></a>&nbsp;&nbsp;<a href="https://github.com/CrypteorCapital"><img src="https://avatars.githubusercontent.com/CrypteorCapital?v=4" width="50px" alt="CrypteorCapital" /></a>&nbsp;&nbsp;<a href="https://github.com/sultanmeghji"><img src="https://avatars.githubusercontent.com/sultanmeghji?v=4" width="50px" alt="sultanmeghji" /></a>&nbsp;&nbsp;<a href="https://github.com/jenius-eagle"><img src="https://avatars.githubusercontent.com/jenius-eagle?v=4" width="50px" alt="jenius-eagle" /></a>&nbsp;&nbsp;<a href="https://github.com/josephjacks"><img src="https://avatars.githubusercontent.com/josephjacks?v=4" width="50px" alt="josephjacks" /></a>&nbsp;&nbsp;<a href="https://github.com/pingshian0131"><img src="https://avatars.githubusercontent.com/pingshian0131?v=4" width="50px" alt="pingshian0131" /></a>&nbsp;&nbsp;<a href="https://github.com/AIdevelopersAI"><img src="https://avatars.githubusercontent.com/AIdevelopersAI?v=4" width="50px" alt="AIdevelopersAI" /></a>&nbsp;&nbsp;<a href="https://github.com/ternary5"><img src="https://avatars.githubusercontent.com/ternary5?v=4" width="50px" alt="ternary5" /></a>&nbsp;&nbsp;<a href="https://github.com/ChrisDMT"><img src="https://avatars.githubusercontent.com/ChrisDMT?v=4" width="50px" alt="ChrisDMT" /></a>&nbsp;&nbsp;<a href="https://github.com/AcountoOU"><img src="https://avatars.githubusercontent.com/AcountoOU?v=4" width="50px" alt="AcountoOU" /></a>&nbsp;&nbsp;<a href="https://github.com/chatgpt-prompts"><img src="https://avatars.githubusercontent.com/chatgpt-prompts?v=4" width="50px" alt="chatgpt-prompts" /></a>&nbsp;&nbsp;<a href="https://github.com/Partender"><img src="https://avatars.githubusercontent.com/Partender?v=4" width="50px" alt="Partender" /></a>&nbsp;&nbsp;<a href="https://github.com/Daniel1357"><img src="https://avatars.githubusercontent.com/Daniel1357?v=4" width="50px" alt="Daniel1357" /></a>&nbsp;&nbsp;<a href="https://github.com/KiaArmani"><img src="https://avatars.githubusercontent.com/KiaArmani?v=4" width="50px" alt="KiaArmani" /></a>&nbsp;&nbsp;<a href="https://github.com/zkonduit"><img src="https://avatars.githubusercontent.com/zkonduit?v=4" width="50px" alt="zkonduit" /></a>&nbsp;&nbsp;<a href="https://github.com/fabrietech"><img src="https://avatars.githubusercontent.com/fabrietech?v=4" width="50px" alt="fabrietech" /></a>&nbsp;&nbsp;<a href="https://github.com/scryptedinc"><img src="https://avatars.githubusercontent.com/scryptedinc?v=4" width="50px" alt="scryptedinc" /></a>&nbsp;&nbsp;<a href="https://github.com/coreyspagnoli"><img src="https://avatars.githubusercontent.com/coreyspagnoli?v=4" width="50px" alt="coreyspagnoli" /></a>&nbsp;&nbsp;<a href="https://github.com/AntonioCiolino"><img src="https://avatars.githubusercontent.com/AntonioCiolino?v=4" width="50px" alt="AntonioCiolino" /></a>&nbsp;&nbsp;<a href="https://github.com/Dradstone"><img src="https://avatars.githubusercontent.com/Dradstone?v=4" width="50px" alt="Dradstone" /></a>&nbsp;&nbsp;<a href="https://github.com/CarmenCocoa"><img src="https://avatars.githubusercontent.com/CarmenCocoa?v=4" width="50px" alt="CarmenCocoa" /></a>&nbsp;&nbsp;<a href="https://github.com/bentoml"><img src="https://avatars.githubusercontent.com/bentoml?v=4" width="50px" alt="bentoml" /></a>&nbsp;&nbsp;<a href="https://github.com/merwanehamadi"><img src="https://avatars.githubusercontent.com/merwanehamadi?v=4" width="50px" alt="merwanehamadi" /></a>&nbsp;&nbsp;<a href="https://github.com/vkozacek"><img src="https://avatars.githubusercontent.com/vkozacek?v=4" width="50px" alt="vkozacek" /></a>&nbsp;&nbsp;<a href="https://github.com/ASmithOWL"><img src="https://avatars.githubusercontent.com/ASmithOWL?v=4" width="50px" alt="ASmithOWL" /></a>&nbsp;&nbsp;<a href="https://github.com/tekelsey"><img src="https://avatars.githubusercontent.com/tekelsey?v=4" width="50px" alt="tekelsey" /></a>&nbsp;&nbsp;<a href="https://github.com/GalaxyVideoAgency"><img src="https://avatars.githubusercontent.com/GalaxyVideoAgency?v=4" width="50px" alt="GalaxyVideoAgency" /></a>&nbsp;&nbsp;<a href="https://github.com/wenfengwang"><img src="https://avatars.githubusercontent.com/wenfengwang?v=4" width="50px" alt="wenfengwang" /></a>&nbsp;&nbsp;<a href="https://github.com/rviramontes"><img src="https://avatars.githubusercontent.com/rviramontes?v=4" width="50px" alt="rviramontes" /></a>&nbsp;&nbsp;<a href="https://github.com/indoor47"><img src="https://avatars.githubusercontent.com/indoor47?v=4" width="50px" alt="indoor47" /></a>&nbsp;&nbsp;<a href="https://github.com/ZERO-A-ONE"><img src="https://avatars.githubusercontent.com/ZERO-A-ONE?v=4" width="50px" alt="ZERO-A-ONE" /></a>&nbsp;&nbsp;</p>



## 🚀 Features

- 🌐 Internet access for searches and information gathering
- 💾 Long-term and short-term memory management
- 🧠 GPT-4 instances for text generation
- 🔗 Access to popular websites and platforms
- 🗃️ File storage and summarization with GPT-3.5
- 🔌 Extensibility with Plugins

## Quickstart

0. Check out the [wiki](https://github.com/Significant-Gravitas/Auto-GPT/wiki)
1. Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
2. Download the [latest release](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest)
3. Follow the [installation instructions][docs/setup]
4. Configure any additional features you want, or install some [plugins][docs/plugins]
5. [Run][docs/usage] the app

Please see the [documentation][docs] for full setup instructions and configuration options.

[docs]: https://docs.agpt.co/

## 📖 Documentation
* [⚙️ Setup][docs/setup]
* [💻 Usage][docs/usage]
* [🔌 Plugins][docs/plugins]
* Configuration
  * [🔍 Web Search](https://docs.agpt.co/configuration/search/)
  * [🧠 Memory](https://docs.agpt.co/configuration/memory/)
  * [🗣️ Voice (TTS)](https://docs.agpt.co/configuration/voice/)
  * [🖼️ Image Generation](https://docs.agpt.co/configuration/imagegen/)

[docs/setup]: https://docs.agpt.co/setup/
[docs/usage]: https://docs.agpt.co/usage/
[docs/plugins]: https://docs.agpt.co/plugins/

## ⚠️ Limitations

This experiment aims to showcase the potential of GPT-4 but comes with some limitations:

1. Not a polished application or product, just an experiment
2. May not perform well in complex, real-world business scenarios. In fact, if it actually does, please share your results!
3. Quite expensive to run, so set and monitor your API key limits with OpenAI!

## 🛡 Disclaimer

This project, Auto-GPT, is an experimental application and is provided "as-is" without any warranty, express or implied. By using this software, you agree to assume all risks associated with its use, including but not limited to data loss, system failure, or any other issues that may arise.

The developers and contributors of this project do not accept any responsibility or liability for any losses, damages, or other consequences that may occur as a result of using this software. You are solely responsible for any decisions and actions taken based on the information provided by Auto-GPT.

**Please note that the use of the GPT-4 language model can be expensive due to its token usage.** By utilizing this project, you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.

As an autonomous experiment, Auto-GPT may generate content or take actions that are not in line with real-world business practices or legal requirements. It is your responsibility to ensure that any actions or decisions made based on the output of this software comply with all applicable laws, regulations, and ethical standards. The developers and contributors of this project shall not be held responsible for any consequences arising from the use of this software.

By using Auto-GPT, you agree to indemnify, defend, and hold harmless the developers, contributors, and any affiliated parties from and against any and all claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from your use of this software or your violation of these terms.

## 🐦 Connect with Us on Twitter

Stay up-to-date with the latest news, updates, and insights about Auto-GPT by following our Twitter accounts. Engage with the developer and the AI's own account for interesting discussions, project updates, and more.

- **Developer**: Follow [@siggravitas](https://twitter.com/siggravitas) for insights into the development process, project updates, and related topics from the creator of Entrepreneur-GPT.
- **Entrepreneur-GPT**: Join the conversation with the AI itself by following [@En_GPT](https://twitter.com/En_GPT). Share your experiences, discuss the AI's outputs, and engage with the growing community of users.

We look forward to connecting with you and hearing your thoughts, ideas, and experiences with Auto-GPT. Join us on Twitter and let's explore the future of AI together!

<p align="center">
  <a href="https://star-history.com/#Torantulino/auto-gpt&Date">
    <img src="https://api.star-history.com/svg?repos=Torantulino/auto-gpt&type=Date" alt="Star History Chart">
  </a>
</p>
