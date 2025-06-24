"""
A minimal training script for MADFormer using PyTorch DDP.
"""
import os
import time
import math
import json
import random
import einops
import datetime 
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from functools import partial
from collections import OrderedDict
from datasets import load_from_disk
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True # True makes A100 training a lot faster
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F

from models import MADFormer

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data.to(torch.float32), alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def build_image_transform_imagenet(image_size, mean, std, tokenizer, max_len):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != 'RGB' else img),
        transforms.Lambda(lambda img: img.crop((0, 0, min(img.size), min(img.size)))), 
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True)
    ])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    def transform_image_batch(batch):
        images = [transform(img) for img in batch['jpg']]
        batch['caption'] = [str(class_int) for class_int in batch['cls']]
        tokenized = tokenizer(batch['caption'], padding=False, truncation=False)
        original_lengths = [len(ids) for ids in tokenized['input_ids']]
        input_ids = tokenizer(batch['caption'], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")['input_ids']
        for orig_len, caption in zip(original_lengths, batch['caption']):
            if orig_len > max_len:
                warnings.warn(f"Caption truncated: {caption[:50]}... (original length: {orig_len}, max: {max_len})")
        return {'input_ids': input_ids, 'image': images}
    
    return transform_image_batch

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new MADFormer model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group(backend="nccl", init_method="env://")
    assert args.global_batch_size % (args.per_gpu_batch_size * dist.get_world_size()) == 0, f"Batch size must be divisible by world size * per gpu batch size."
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    seed = args.global_seed * dist.get_world_size() + local_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()} on device cuda:{local_rank}.")

    experiment_index = int(args.id) if args.id else len(glob(f"{args.results_dir}/*"))
    model_string_name = "MADFormer"    
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    model = MADFormer(
        latent_size=args.vae_latent_size*args.patch_size**2,
        block_size=args.block_size,
        ar_len=args.ar_len,
        spatial_len=args.spatial_len,
        square_block=args.square_block,
        model_config_path=args.model_config_path,
        diff_depth=args.diff_depth,
        clear_clean=args.clear_clean,
        clear_cond=args.clear_cond,
        denoising_mlp=args.denoising_mlp
    ).to(local_rank)
    with open(args.model_config_path, "r") as file:
        model_config = json.load(file)
    max_len = model_config.get('max_position_embeddings', None)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(local_rank).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-3.2-1B')
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank).eval()
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision =='fp16'))

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    scheduler = DDPMScheduler()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if args.dataset == 'imagenet':
        dataset = load_from_disk(args.dataset_path)
        dataset.set_transform(build_image_transform_imagenet(args.image_size, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], tokenizer, max_len))
    else:
        raise NotImplementedError

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.per_gpu_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    accumulation_steps = args.global_batch_size // (args.per_gpu_batch_size * dist.get_world_size())
    step_per_epoch = len(dataloader) // accumulation_steps
    total_steps = args.epochs * step_per_epoch
    if args.lr_scheduler == "wsd":
        def lambda_wsd(step, stages, end_lr_exponent=6):
            if step < stages[0]:
                return step /stages[0]
            elif step < stages[1]:
                return 1.0
            else:
                ratio = (step - stages[1])/(stages[2] - stages[1])
                return 0.5**(ratio*end_lr_exponent)
        lrs = lr_scheduler.LambdaLR(opt, lr_lambda=partial(lambda_wsd, stages=[int(args.lr_warmup*total_steps), int(0.85*total_steps), total_steps], end_lr_exponent=6))
    elif args.lr_scheduler == "cosine":
        def lambda_cosine(step, stages, end_lr_ratio=0):
            if step < stages[0]:
                return step /stages[0]
            else:
                ratio = (step - stages[0])/(stages[1] - stages[0])
                return end_lr_ratio + 0.5 * (1 - end_lr_ratio) * (1 + math.cos(ratio*math.pi))
        lrs = lr_scheduler.LambdaLR(opt, lr_lambda=partial(lambda_cosine, stages=[int(args.lr_warmup*total_steps), total_steps], end_lr_ratio=0.01))
    else:
        lrs = None

    logger.info(f"Total steps: {total_steps}")

    # Variables for monitoring/logging purposes:
    running_loss = 0
    start_epoch = 0
    train_steps = 0
    last_train_step = 0
    step_in_current_epoch = 0
    resume_step_in_current_epoch = 0
    start_time = time.time()

    resume_checkpoint_path = f"{checkpoint_dir}/latest.pt"
    logger.info(f"Checking for resume checkpoint at {resume_checkpoint_path}")
    dist.barrier()
    if os.path.exists(resume_checkpoint_path):
        torch.cuda.empty_cache()
        checkpoint = torch.load(resume_checkpoint_path, map_location=f"cuda:{local_rank}", weights_only=False)
        logger.info(f"Loading checkpoint from {resume_checkpoint_path}")
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        model.train()
        ema.eval()
        opt.load_state_dict(checkpoint["opt"])
        if lrs is not None and "lrs" in checkpoint:
            lrs.load_state_dict(checkpoint["lrs"])
        train_steps = checkpoint.get("train_steps", 0)
        start_epoch = checkpoint.get("epoch", 0)
        running_loss = checkpoint.get("running_loss", 0)
        resume_step_in_current_epoch = checkpoint.get("step_in_current_epoch", 0)
        dist.barrier()
        logger.info("="*30)
        logger.info(f"Loaded ckpt from step {train_steps}, epoch {start_epoch}, "
                    f"batch {resume_step_in_current_epoch} in current epoch")
    else:
        logger.info(f"Starting fresh train for {args.epochs} epochs...")
    vae_scaling_factor = vae.config.get("scaling_factor", 0.18215)

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accumulation_counter = 0
        step_in_current_epoch = 0
        
        dataloader_iterator = iter(dataloader)
        num_data_to_skip = resume_step_in_current_epoch * accumulation_steps + accumulation_counter
        if epoch == start_epoch and num_data_to_skip > 0:
            for _ in range(num_data_to_skip):
                next(dataloader_iterator)
            last_train_step = train_steps
            step_in_current_epoch = resume_step_in_current_epoch
            logger.info(f"Resuming train at epoch {epoch}, step {resume_step_in_current_epoch} and accum step {accumulation_counter}...")
        
        for data in dataloader_iterator:
            input_ids = data['input_ids'].to(local_rank)     
            x = data['image'].to(local_rank)
            batch_size = x.size(0)
            with torch.inference_mode():
                # Map input images to latent space + normalize latents:
                image_shape = x.shape[-3:]
                latent = vae.encode(x.view(-1, *image_shape)).latent_dist.sample().mul_(vae_scaling_factor)
                _, _, latent_h, latent_w = latent.size()
                latent = einops.rearrange(latent, 'N C (h1 h2) (w1 w2) -> N (h1 w1) (h2 w2 C)', h2=args.patch_size, w2=args.patch_size)
            latent = latent.clone() # other wise it is an inference tensor

            if args.square_block:
                assert math.isqrt(args.pre_downsample_block_size) ** 2 == args.pre_downsample_block_size
                latent = einops.rearrange(latent, 'N (H W) C-> N H W C', H=latent_h//args.patch_size, W=latent_h//args.patch_size)
                block_h = block_w = int(math.isqrt(args.pre_downsample_block_size))
                latent = einops.rearrange(latent, 'N (new_h block_h) (new_w block_w) c -> N new_h new_w block_h block_w c', block_h=block_h, block_w=block_w)
                latent = einops.rearrange(latent, '(N T) new_h new_w block_h block_w c -> N (T new_h new_w) (block_h block_w) c', T=args.num_frames)
            else:
                latent = einops.rearrange(latent, '(N T) B C -> N T B C', N=batch_size)
                latent = einops.rearrange(latent, 'N T B C -> N (T B) C')
                latent = einops.rearrange(latent, 'N (T B) C-> N T B C', B=args.block_size) # For multi-frames
            
            N, T, B, C = latent.shape
            noise = torch.randn_like(latent)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size, T))
            noised_latent = scheduler.add_noise(latent.reshape(-1, 1, B, C), noise.reshape(-1, 1, B, C), t.reshape(-1)).reshape(N, T, B, C)

            with torch.amp.autocast('cuda', dtype=ptdtype):
                noise_pred, condition, clean_tower_output, logits = model(input_ids, latent, noised_latent, t)
                if args.text_loss_lambda != 0:
                    logits = logits[..., :-1, :].contiguous().reshape(-1, logits.shape[-1])
                    labels = input_ids[..., 1:].contiguous().reshape(-1)
                    text_loss = F.cross_entropy(logits, labels)
                else:
                    text_loss = 0
                noise_pred = einops.rearrange(noise_pred, '(N T) C H W -> N T (H W) C', T=T)
                condition = einops.rearrange(condition, '(N T) C H W -> N T (H W) C', T=T)
                image_loss = F.mse_loss(noise_pred, noise)
                hidden_loss = F.mse_loss(condition, latent) if not args.clear_cond else 0
                if args.clear_clean and args.clear_tower_loss_lambda != 0 and clean_tower_output is not None:
                    clean_tower_output = einops.rearrange(clean_tower_output, '(N T) C H W -> N T (H W) C', T=T)[:, :-1, :, :]
                    clear_tower_loss = F.mse_loss(clean_tower_output, latent[:, :-1, :, :])
                else:
                    clear_tower_loss = 0
                loss_orig = args.text_loss_lambda * text_loss + args.image_loss_lambda * image_loss + args.lambda_hidden_loss * hidden_loss + args.clear_tower_loss_lambda * clear_tower_loss
                loss = loss_orig / accumulation_steps
            
            scaler.scale(loss).backward()
            accumulation_counter += 1
            running_loss += loss_orig.item()
            
            if accumulation_counter == accumulation_steps:
                scaler.unscale_(opt)
                if args.max_grad_norm != 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
                else:
                    grad_norm = 0.0

                scaler.step(opt)
                scaler.update()
                if lrs is not None:
                    lrs.step()
                update_ema(ema, model.module)
                opt.zero_grad()
                accumulation_counter = 0
                train_steps += 1
                step_in_current_epoch +=  1

                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = (train_steps - last_train_step) / (end_time - start_time)
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / (train_steps * accumulation_steps), device=local_rank)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, LR: {opt.param_groups[0]['lr']}, "
                                f"Grad Norm: {grad_norm:.4f}, Text Loss: {text_loss:.4f}, Hidden Loss: {hidden_loss:.4f}, Image Loss: {image_loss:.4f}, Clean Tower Loss: {clear_tower_loss:.4f} "
                                f"Time: {current_time}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    last_train_step = train_steps
                    running_loss = 0
                    start_time = time.time()

                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "lrs": lrs.state_dict() if lrs is not None else {},
                            "args": args,
                            "train_steps": train_steps,
                            "step_in_current_epoch": step_in_current_epoch,
                            "running_loss": running_loss,
                            "epoch": epoch
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        torch.save(checkpoint, f"{checkpoint_dir}/latest.pt")
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if accumulation_counter > 0:
            scaler.unscale_(opt)
            if args.max_grad_norm != 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
            else:
                grad_norm = 0.0

            scaler.step(opt)
            scaler.update()
            if lrs is not None:
                lrs.step()
            update_ema(ema, model.module)
            opt.zero_grad()
            train_steps += 1
            accumulation_counter = 0
        
        dist.barrier()

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, choices=['imagenet'], default='imagenet')
    parser.add_argument("--dataset_path", type=str, default='./datasets/imagenet')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--per_gpu_batch_size", type=int, default=4)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--mixed_precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--image_size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--patch_size", default=1, type=int, help="patch_size x patch_size of vae latents forms an input feature")
    parser.add_argument("--vae_latent_size", default=4, type=int, help="vae's latent feature size")
    parser.add_argument("--block_size", default=64, type=int, help="vae's latent feature size")
    parser.add_argument("--pre_downsample_block_size", default=256, type=int, help="vae's latent feature size")
    parser.add_argument("--vae_patch_pixels", default=8, type=int, help="sqrt of pixels in a vae token")
    parser.add_argument("--square_block", action="store_true", default=True)
    parser.add_argument("--ar_len", type=int, default=4)
    parser.add_argument("--use_rope", action="store_true", default=True)
    parser.add_argument("--lr_scheduler", type=str, default="wsd", choices=["wsd", "cosine", "constant"])
    parser.add_argument("--lr_warmup", type=float, default=0.01)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_config_path", type=str, default="./src/configs/model/madformer.json")
    parser.add_argument("--spatial_len", type=int, default=256)
    parser.add_argument("--lambda_hidden_loss", type=float, default=0.1)
    parser.add_argument("--text_loss_lambda", type=float, default=0.0)
    parser.add_argument("--image_loss_lambda", type=float, default=1.0)
    parser.add_argument("--diff_depth", type=int, default=14)
    parser.add_argument("--clear_clean", action="store_true", default=False)
    parser.add_argument("--clear_cond", action="store_true", default=False)
    parser.add_argument("--denoising_mlp", action="store_true", default=False)
    parser.add_argument("--clear_tower_loss_lambda", type=float, default=0.0)
    parser.add_argument("--single_tower", action="store_true", default=False)
    parser.add_argument("--id", type=str, default=None)
    args = parser.parse_args()

    print(args)
    main(args)
