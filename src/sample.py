import os
import torch
import numpy as np
import json
import math
import einops
import random
import argparse
from tqdm import tqdm
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, AutoencoderKL
from torchvision.utils import save_image
from transformers import AutoTokenizer
from models import MADFormer
from config import MADFormerConfig

def sample_on_single_gpu(model_ckpt, vae_path, model_config_path, sample_step, save_dir, args_model, range_start, range_end, gen_num, class_num, max_bs, model_config, tokenizer):
    """Runs inference on a single GPU, using batch processing."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model on the single GPU
    model = MADFormer(
        latent_size=args_model.vae_latent_size * args_model.patch_size ** 2,
        block_size=args_model.block_size,
        ar_len=args_model.ar_len,
        spatial_len=args_model.spatial_len,
        square_block=True,
        model_config_path=model_config_path if model_config_path else args_model.model_config_path,
        diff_depth=args_model.diff_depth,
        clear_cond=getattr(args_model, 'clear_cond', False),
        clear_clean=getattr(args_model, 'clear_clean', False),
        denoising_mlp=getattr(args_model, 'denoising_mlp', False)
    ).to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    scheduler = DDIMScheduler()
    sample_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    vae_scaling_factor = vae.config.get("scaling_factor", 0.18215)

    # Calculate number of images per class
    images_per_class = gen_num // class_num
    print(f"Generating {images_per_class} images per class")
    num_samples = range_end - range_start + 1
    
    # Initialize batches
    num_batches = (num_samples + max_bs - 1) // max_bs  # Round up
    print(f"Processing {num_samples} samples in {num_batches} batches.")

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * max_bs
        end_idx = min((batch_idx + 1) * max_bs, num_samples)
        batch_size = end_idx - start_idx
        
        # Determine which classes are needed for this batch
        # Calculate global indices first (with offset)
        global_indices = [range_start + start_idx + i for i in range(batch_size)]
        
        # For each global index, determine which class it belongs to
        class_assignments = [str(idx // images_per_class) for idx in global_indices]
        
        # Tokenize the class assignments (prompts)
        tokenized_batch = tokenizer(class_assignments, padding="max_length", truncation=True, 
                             max_length=model_config.max_position_embeddings, return_tensors="pt")['input_ids'].to(device)
        
        # Run inference
        with torch.no_grad():
            output = model.sample(
                tokenized_batch,
                sample_scheduler, sample_step,
                target_shape=(batch_size, args_model.ar_len, args_model.pre_downsample_block_size, 
                             args_model.vae_latent_size * args_model.patch_size ** 2),
                dtype=torch.float
            )
            block_h = block_w = int(math.isqrt(args_model.pre_downsample_block_size))
            new_h = new_w = int(math.isqrt(output.size(1))) 
            latent = einops.rearrange(output, 'b (new_h new_w) (block_h block_w) c -> b new_h new_w block_h block_w c',
                                     block_h=block_h, block_w=block_w, new_h=new_h, new_w=new_w)
            latent = einops.rearrange(latent, 'b new_h new_w block_h block_w c -> b (new_h block_h new_w block_w) c')
            latent = einops.rearrange(latent, 'N (h1 w1) (h2 w2 C) -> N C (h1 h2) (w1 w2)',
                                     h1=(args_model.image_size // 8) // args_model.patch_size, 
                                     w1=(args_model.image_size // 8) // args_model.patch_size,
                                     h2=args_model.patch_size, w2=args_model.patch_size)
            samples = vae.decode(latent / vae_scaling_factor).sample

        # Save images with proper indices
        for i, sample in enumerate(samples):
            global_idx = range_start + start_idx + i
            save_path = os.path.join(save_dir, f"sample_img_{global_idx}.png")
            save_image(sample, save_path, normalize=True, value_range=(-1, 1))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--sample_step", type=int, default=100)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--gen_num", type=int, default=100, help="Total number of images to generate")
    parser.add_argument("--class_num", type=int, default=1000, help="Total number of classes to generate (1000 for imagenet)")
    parser.add_argument("--range_start", type=int, default=0, help="Starting index for image generation")
    parser.add_argument("--range_end", type=int, default=-1, help="Ending index for image generation (-1 means use all)")
    parser.add_argument("--max_bs", type=int, default=16, help="Maximum batch size for generation")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    args_model = ckpt['args']
    model_ckpt = ckpt['model']
    vae_path = 'stabilityai/sd-vae-ft-mse'
    model_config = MADFormerConfig(**json.load(open(args.model_config_path, "r"))) if args.model_config_path else MADFormerConfig(**json.load(open(args_model.model_config_path, "r")))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Handle range specifications
    effective_range_end = args.range_end if args.range_end != -1 else args.gen_num - 1
    
    if args.range_start > effective_range_end:
        print(f"Error: range_start ({args.range_start}) is greater than range_end ({effective_range_end})")
        exit(1)
    
    # Calculate how many images to generate in this range
    range_size = effective_range_end - args.range_start + 1
    print(f"Generating images in range {args.range_start} to {effective_range_end} (total: {range_size})")
    
    # Process everything on a single GPU
    sample_on_single_gpu(
        model_ckpt,
        vae_path,
        args.model_config_path,
        args.sample_step,
        args.save_dir,
        args_model,
        args.range_start,
        effective_range_end,
        args.gen_num,
        args.class_num,
        args.max_bs,
        model_config,
        tokenizer
    )

    print(f"Finished generating images for range {args.range_start} to {effective_range_end}")