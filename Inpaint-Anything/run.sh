python remove_anything.py \
    --input_img    /home/tmahmud/diffusion/MultiConcept_Diffusion/CustomData/cat_and_dog/IMG_4218.jpg \
    --coords_type key_in \
    --point_coords 500 200 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results/CustomData/dog/IMG_4218 \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama

    # --input_img ./data/cat/cat3.png \

# python remove_anything.py \
#     --input_img    /home/tmahmud/diffusion/MultiConcept_Diffusion/CustomData/cat_and_dog/IMG_4218.jpg \
#     --coords_type key_in \
#     --point_coords 200 300 \
#     --point_labels 1 \
#     --dilate_kernel_size 15 \
#     --output_dir ./results/CustomData/cat/IMG_4218 \
#     --sam_model_type "vit_h" \
#     --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
#     --lama_config ./lama/configs/prediction/default.yaml \
#     --lama_ckpt ./pretrained_models/big-lama