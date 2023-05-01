"""
Readme:

Step 1-- Git clone https://github.com/YuxinWenRick/hard-prompts-made-easy
Step 2-- Copy this file within that repo
Step 3-- Sample run command python run_evaluation-hp.py <dir from where to source images> <dir where generated images will be stored> <prompt with v*s...eg: "A photo of v* cat" (without quotes)>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import glob
import sys
from PIL import Image
import torch

import open_clip
import argparse
from optim_utils import *
import mediapy as media

print("Initializing...")
config_path = "sample_config.json"

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Open image files and store in variables
"""

"""
Specify the source and generated image directories
"""
source_path_dir=sys.argv[1]
generated_path_dir=sys.argv[2]
  

"""
Open image files and store in variables
"""
print(f"============================Load source images============================================")
print(f"Loding source files from {source_path_dir}...")
source_images = [Image.open(image_path) for image_path in glob.iglob(f'{source_path_dir}/*') 
          if (image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"))]
os.makedirs(generated_path_dir,exist_ok=True)
print(f"============================================================================")

clip_model="ViT-H-14"
clip_pretrain="laion2b_s32b_b79k"

"""
Prompt Optimization via CLIP
"""
print(f"============================Prompt Optimization============================================")

text_provided= " ".join(sys.argv[3:])
print("provided text--->", text_provided)

args.print_new_best = True
model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrain, device=device)

print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
    print(f"Intermediate results will be printed every {args.print_step} steps.")

# optimize prompt
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=source_images,preprovided_text=text_provided)
print("The learned Prompt is--->",learned_prompt)
print(f"============================================================================")


"""
Generate 4 images using optimized prompt and save them in a folder
"""
print(f"===============SETUP Diffusion Model for Image generation===================")

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision="fp16",
    )
image_length = 512
pipe = pipe.to(device)
print(f"============================================================================")

print(f"============================Image Generation using prompts====================")
prompt=learned_prompt
num_images = 4
guidance_scale = 9
num_inference_steps = 25

gen_images = pipe(
    prompt,
    num_images_per_prompt=num_images,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    height=image_length,
    width=image_length,
    ).images


for idx,gen_image in enumerate(gen_images):
    media.write_image(f"{generated_path_dir}/{idx}.png",image=gen_image)
print(f"=============================================================================")

"""
#####################################################################################
#####################################################################################
QUANT. EVALUATION CODE
#####################################################################################
#####################################################################################
"""

"""
Load models
"""
clip_model_neutr="ViT-g-14"
clip_pretrain_neutr="laion2b_s12b_b42k"
print(f"=============================MODEL===========================================")
print(f"Loding the (Neutral) clip model {clip_model_neutr}...")
model_netr, _, preprocess_neutr = open_clip.create_model_and_transforms(clip_model_neutr, pretrained=clip_pretrain_neutr, device=device)
print(f"=============================================================================")



print(f"============================Load generated images===========================")
print(f"Loding generated files from {generated_path_dir}...")
generated_images = [Image.open(image_path) for image_path in glob.iglob(f'{generated_path_dir}/*') 
          if (image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"))]
print(f"============================================================================")



def measure_similarity(source_images, generated_images, ref_model, ref_clip_preprocess, device="cuda",
                      option="mean,cos"):
    """
    :params
    option : consists of three words x,y,z
                mean,cos--- first normalize embeddings then take mean separately of source 
                                 and generated images and then take cosine similarity
                cos,mean--- first take cosine similarity between the embeddings every source image 
                            vs every generated image and then take the average.
    """
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in source_images]
        ori_batch = torch.cat(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in generated_images]
        gen_batch = torch.cat(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)

        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        if option=="mean,cos":
            mean_source_feat=torch.mean(ori_feat,dim=0)
            mean_generated_feat=torch.mean(gen_feat,dim=0)
            mean_source_feat = mean_source_feat / mean_source_feat.norm()
            mean_generated_feat = mean_generated_feat / mean_generated_feat.norm()
            return torch.dot(mean_source_feat,mean_generated_feat).item()
        
        elif option=="cos,mean":
            cross_similarity=(ori_feat @ gen_feat.t()).mean().item()
            return cross_similarity
        
print(f"===============================EVALUATION SCORES=========================================")
print("(Mean,cosine) similarity is %.3f"%measure_similarity(source_images,generated_images,model_netr,preprocess_neutr,option="mean,cos"))
print("(cosine,Mean) similarity is %.3f"%measure_similarity(source_images,generated_images,model_netr,preprocess_neutr,option="cos,mean"))
print(f"=========================================================================================")

