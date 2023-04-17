MODEL_NAME="CompVis/stable-diffusion-v1-4"

accelerate launch src/diffusers_training_clip_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/cat  \
          --class_data_dir=./real_reg/samples_cat/ \
          --output_dir=./logs/clip_lora/cat  \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> cat"  \
          --class_prompt="cat" \
          --train_text_encoder \
          --resolution=224  \
          --train_batch_size=4  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=250 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>" \
          --use_lora \
          --lora_r 16 \
          --lora_alpha 27 \
          --lora_text_encoder_r 16 \
          --lora_text_encoder_alpha 17

# python src/diffusers_sample_lora.py --delta_ckpt logs/clip_lora/cat/ --ckpt "logs/clip_lora/cat/" --prompt "<new1> cat playing with a ball"