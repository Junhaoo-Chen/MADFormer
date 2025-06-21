import os
import torch
import numpy as np
import math
import einops
import random
import argparse
from tqdm import tqdm
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoencoderKL
from torchvision.utils import save_image
from models import MADFormer

def sample_on_single_gpu(model_ckpt, vae_path, sample_step, save_dir, args, cache_clean, range_start, range_end, max_bs):
    """Runs inference on a single GPU, using batch processing."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MADFormer(
        latent_size=args.vae_latent_size * args.patch_size ** 2,
        block_size=args.block_size,
        ar_len=args.ar_len,
        spatial_len=args.spatial_len,
        square_block=True,
        model_config_path=args.model_config_path,
        diff_depth=args.diff_depth,
        clear_cond=getattr(args, 'clear_cond', False),
        clear_clean=getattr(args, 'clear_clean', False),
        denoising_mlp=getattr(args, 'denoising_mlp', False),
        cache_clean=cache_clean
    ).to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    scheduler = DDIMScheduler()
    sample_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    vae_scaling_factor = vae.config.get("scaling_factor", 0.18215)

    # Create labels for the range we're generating
    num_samples = range_end - range_start + 1
    labels = torch.tensor([0] * num_samples).to(device)
    
    num_batches = (num_samples + max_bs - 1) // max_bs  # Round up
    print(f"Processing {num_samples} samples in {num_batches} batches.")

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * max_bs
        end_idx = min((batch_idx + 1) * max_bs, num_samples)
        batch_labels = labels[start_idx:end_idx]
        
        with torch.no_grad():
            output = model.sample(
                sample_scheduler, sample_step,
                target_shape=(batch_labels.size(0), args.ar_len, args.pre_downsample_block_size, args.vae_latent_size * args.patch_size ** 2),
                dtype=torch.float
            )
            block_h = block_w = int(math.isqrt(args.pre_downsample_block_size))
            new_h = new_w = int(math.isqrt(output.size(1)))
            latent = einops.rearrange(output, 'b (new_h new_w) (block_h block_w) c -> b new_h new_w block_h block_w c',
                                      block_h=block_h, block_w=block_w, new_h=new_h, new_w=new_w)
            latent = einops.rearrange(latent, 'b new_h new_w block_h block_w c -> b (new_h block_h new_w block_w) c')
            latent = einops.rearrange(latent, 'N (h1 w1) (h2 w2 C) -> N C (h1 h2) (w1 w2)',
                                      h1=(args.image_size // 8) // args.patch_size, w1=(args.image_size // 8) // args.patch_size,
                                      h2=args.patch_size, w2=args.patch_size)
            samples = vae.decode(latent / vae_scaling_factor).sample

        for i, sample in enumerate(samples):
            global_idx = range_start + start_idx + i
            save_path = os.path.join(save_dir, f"sample_img_{global_idx}.png")
            save_image(sample, save_path, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--sample_step", type=int, default=25)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--cache_clean", action="store_true", default=False)
    parser.add_argument("--range_start", type=int, default=0, help="Starting index for image generation")
    parser.add_argument("--range_end", type=int, default=7, help="Ending index for image generation (-1 means use all)")
    parser.add_argument("--max_bs", type=int, default=16, help="Batch size per gpu for generation")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    args_model = ckpt['args']
    model_ckpt = ckpt['model']
    vae_path = 'stabilityai/sd-vae-ft-mse'

    if args.range_start > args.range_end:
        print(f"Error: range_start ({args.range_start}) is greater than range_end ({args.range_end})")
        exit(1)
    
    # Calculate how many images to generate in this range
    range_size = args.range_end - args.range_start + 1
    print(f"Generating images in range {args.range_start} to {args.range_end} (total: {range_size})")
    
    # Process everything on a single GPU
    sample_on_single_gpu(
        model_ckpt, 
        vae_path, 
        args.sample_step, 
        args.save_dir, 
        args_model, 
        args.cache_clean, 
        args.range_start, 
        args.range_end,
        args.max_bs
    )

    print(f"Finished generating images for range {args.range_start} to {args.range_end}")