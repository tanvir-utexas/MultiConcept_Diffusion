# ==========================================================================================
#
# MIT License. To view a copy of the license, visit MIT_LICENSE.md.
#
# ==========================================================================================

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig

sys.path.append('./')
from src.diffusers_model_pipeline import CustomDiffusionPipeline

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    # pipe = StableDiffusionPipeline.from_pretrained(
    #     base_model_name_or_path, torch_dtype=dtype, requires_safety_checker=False
    # ).to(device)
    pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
    # pipe.load_model("logs/clip_sd/cat/delta.bin", None)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe


def sample(ckpt, delta_ckpt, from_file, prompt, compress, batch_size, freeze_model):
    model_id = ckpt
    pipe = get_lora_sd_pipeline(os.path.join(model_id), adapter_name="cat")

    # pipe = CustomDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # pipe.load_model(delta_ckpt, compress)

    outdir = os.path.dirname(delta_ckpt)
    generator = torch.Generator(device='cuda').manual_seed(42)

    all_images = []
    if prompt is not None:
        images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1., generator=generator).images
        all_images += images
        images = np.hstack([np.array(x) for x in images])
        images = Image.fromarray(images)
        # takes only first 50 characters of prompt to name the image file
        name = '-'.join(prompt[:50].split())
        images.save(f'{outdir}/{name}.png')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [[prompt]*batch_size for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1., generator=generator).images
            all_images += images
            images = np.hstack([np.array(x) for x in images], 0)
            images = Image.fromarray(images)
            # takes only first 50 characters of prompt to name the image file
            name = '-'.join(prompt[0][:50].split())
            images.save(f'{outdir}/{name}.png')

    os.makedirs(f'{outdir}/samples', exist_ok=True)
    for i, im in enumerate(all_images):
        im.save(f'{outdir}/samples/{i}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default='./',
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.batch_size, args.freeze_model)
