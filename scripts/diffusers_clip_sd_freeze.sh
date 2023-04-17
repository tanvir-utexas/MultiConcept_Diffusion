MODEL_NAME="CompVis/stable-diffusion-v1-4"

accelerate launch src/diffusers_training_clip_sd_freeze.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/cat  \
          --class_data_dir=./real_reg/samples_cat/ \
          --output_dir=./logs/clip_sd_freeze/cat_4GPU  \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> cat"  \
          --class_prompt="cat" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=250 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py --delta_ckpt logs/clip_sd/cat_1GPU/delta.bin --ckpt "CompVis/stable-diffusion-v1-4" --prompt "<new1> cat with a dog in times square"