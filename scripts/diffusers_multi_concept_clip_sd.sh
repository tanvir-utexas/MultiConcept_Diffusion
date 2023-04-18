MODEL_NAME="CompVis/stable-diffusion-v1-4"

CUDA_VISIBLE_DEVICES=2 accelerate launch src/diffusers_training_clip_sd.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs/CustomData/cat_and_dog_clip_sd_no_cmn_img  \
          --concepts_list=./CustomData/concept_list_2.json \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>+<new2>"

# python src/diffusers_sample.py --delta_ckpt ./logs/CustomData/cat_and_dog_clip_sd/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "<new1> cat in times square"
