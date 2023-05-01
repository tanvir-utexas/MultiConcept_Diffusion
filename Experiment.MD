Follow first Readme.md for installation steps. 
These are the list of experiments we ran. 
```bash
##For changing data location in any training script, change the instance_data_dir accordingly.
##For text prompting for multi-concept images using our framework, run training first by using the follwing command, change flag concepts_list=./CustomData2/concept_list.json in scripts/diffusers_multi_concept_clip_sd.sh and then run: 
sh scripts/diffusers_multi_concept_clip_sd.sh
##For text prompting for multi-concept images using our framework for inference run, please change delta_ckpt below, depending on the outputdir flag set in the previous command, as that is the location where the checkpoint will be stored:
python src/diffusers_sample.py --delta_ckpt ./logs/CustomData/cat_and_dog_clip_sd/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "a <X*> cat and a <Y#> dog in swimming pool"
##For text prompting for single concept image with the cropped cat and dog images, run training first by using the follwing command, change flag concepts_list=./CustomData2/concept_list2.json in scripts/diffusers_multi_concept_clip_sd.sh and then run: 
sh scripts/diffusers_multi_concept_clip_sd.sh
#For text prompting for multi-concept images inference run, please change delta_ckpt below, depending on the outputdir flag set in the previous command, as that is the location where the checkpoint will be stored:
python src/diffusers_sample.py --delta_ckpt ./logs/CustomData/cat_and_dog_clip_sd/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "a <X*> cat and a <Y#> dog in swimming pool"
##For running single concept image with our framework, run training first: 
sh scripts/diffusers_clip_sd.sh
##For running single concept image in inference with our framework run, please change delta_ckpt below, depending on the outputdir flag set in the previous command, as that is the location where the checkpoint will be stored:
python src/diffusers_sample.py --delta_ckpt ./logs/CustomData/cat_sd/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "a <new1> cat in swimming pool"
##Note: The inference command is same with difference in the value to be provided. Prompt flag represent the output you want to provide 
##For other ablation study, putting only the training commands ahead, use the same inference commands:
##For running single concept image using CustomDiffusion
sh scripts/diffusers_single.sh
##For running single concept image using clip
sh scripts/diffusers_clip.sh
##For running single concept image using our framework but with frozen custom diffusion, run:
sh scripts/diffusers_clip_sd_freeze.sh
##For running single concept image using our framework band LoRA, run:
sh dreanbooth_lora.sh
##For running inference with  our framework band LoRA, run:
python src/diffusers_sample_lora.py --delta_ckpt ./logs/CustomData/cat_sd/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "a <new1> cat in swimming pool"
##Please check the flags, if you want to change anything in the run. 
##For evaluating hard prompts, pls follow run_evaluation-hp.py
```