import json
import math
import torch
import einops

import torch.nn as nn
import numpy as np
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor
from transformers.activations import ACT2FN
from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D, DownBlock2D, AttnUpBlock2D, UpBlock2D

from rope import RopeND
from config import MADFormerConfig

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    flex_attention = torch.compile(flex_attention)
    from torch.nn.attention.flex_attention import create_block_mask
    from functools import partial
    USE_FLEX_ATTENTION = True
    print("Use flex attention!!!!")
except:
    USE_FLEX_ATTENTION = False
    print('Not use flex attention')

torch._dynamo.config.optimize_ddp=False

#################################################################################
#           Embedding Layers for Timesteps and UNet Up/DownSamplers             #
#################################################################################
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = self.get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)
        self.reset_parameters()
    
    def get_activation(self, act_fn: str):
        ACTIVATION_FUNCTIONS = {
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }
        act_fn = act_fn.lower()
        if act_fn in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[act_fn]
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=std)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.linear_1(x.type_as(self.linear_1.weight))
        x = self.act(x)
        x = self.linear_2(x)
        return x
    
class UNetDownsampler(nn.Module):
    """
    UNet downsampling module with ResNet and Transformer blocks.

    Parameters:
        config (MADFormerConfig): Configuration object with the following attributes:
            - unet_block_out_channels: List of output channels per block.
            - unet_num_layers_per_block: List of ResNet layers per block.
            - unet_resnet_groups: Number of groups in ResNet layers.
            - unet_transformer_layers_per_block: Transformer layers per block.
            - unet_num_heads: Number of attention heads for transformer blocks.
            - latent_patch_dim: Input dimension for the latent patch.
            - unet_time_emb_dim: Time embedding dimension.
            - hidden_size: Output dimension after the final projection.
                
    Output:
        Tensor of shape [B, H'', W'', hidden_size]
    """

    def __init__(self, config: MADFormerConfig):
        super().__init__()

        in_channels = config.latent_patch_dim
        self.conv_in = nn.Conv2d(in_channels, config.unet_block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.conv_out = nn.Conv2d(config.unet_block_out_channels[-1], config.hidden_size, kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(
            in_channels=1,
            time_embed_dim=config.unet_time_emb_dim,
            act_fn="silu"
        )

        in_channels = config.unet_block_out_channels[0]
        # Create downsampling blocks
        for i, (out_channels, num_layers, transformer_layers) in enumerate(
            zip(config.unet_block_out_channels, config.unet_num_layers_per_block, config.unet_transformer_layers_per_block)
        ):
            if transformer_layers == 1:
                # Use AttnDownBlock2D without downsampling layer
                block = AttnDownBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=config.unet_time_emb_dim,
                    num_layers=num_layers,
                    resnet_groups=config.unet_resnet_groups,
                    attention_head_dim=out_channels // config.unet_num_heads,
                    downsample_padding=1,
                    downsample_type='conv' if (i < len(config.unet_block_out_channels) - 1) else None,
                )
            else:
                # Use DownBlock2D for 0 transformer layers
                block = DownBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=config.unet_time_emb_dim,
                    num_layers=num_layers,
                    resnet_groups=config.unet_resnet_groups,
                    downsample_padding=1,
                    add_downsample=(i < len(config.unet_block_out_channels) - 1),
                )

            self.down_blocks.append(block)
            in_channels = out_channels

    def forward(
        self, hidden_states, time_steps
    ):
        temb = self.time_embedding(time_steps.reshape(-1, 1))
        hidden_states = self.conv_in(hidden_states)
        
        output_states = (hidden_states, )
        for block in self.down_blocks:
            hidden_states, output = block(hidden_states=hidden_states, temb=temb)
            output_states += output

        hidden_states = self.conv_out(hidden_states)

        return hidden_states, output_states
    
class UNetUpsampler(nn.Module):
    """
    UNet upsampling module with ResNet and Transformer blocks.

    Parameters:
        config (MADFormerConfig): Configuration object with the following attributes:
            - unet_block_out_channels: List of output channels per block.
            - unet_num_layers_per_block: List of ResNet layers per block.
            - unet_resnet_groups: Number of groups in ResNet layers.
            - unet_transformer_layers_per_block: Transformer layers per block.
            - unet_num_heads: Number of attention heads for transformer blocks.
            - latent_patch_dim: Input dimension for the latent patch.
            - unet_time_emb_dim: Time embedding dimension.
            - hidden_size: Output dimension after the final projection.
                
    Output:
        Tensor of shape [B, H, W, latent_patch_dim]
    """
    
    def __init__(self, config: MADFormerConfig):
        super().__init__()

        self.up_blocks = nn.ModuleList()
        self.conv_in = nn.Conv2d(config.hidden_size, config.unet_block_out_channels[-1], kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(
            in_channels=1,
            time_embed_dim=config.unet_time_emb_dim,
            act_fn="silu"
        )
        self.conv_out = nn.Conv2d(config.unet_block_out_channels[0], config.latent_patch_dim, kernel_size=3, padding=1)
        
        reversed_block_out_channels = list(reversed(config.unet_block_out_channels))
        reversed_num_layers_per_block = list(reversed(config.unet_num_layers_per_block))
        reversed_transformer_layers_per_block = list(reversed(config.unet_transformer_layers_per_block))
        output_channel = reversed_block_out_channels[0]
        # Create upsampling blocks
        for i, (out_channels, num_layers, transformer_layers) in enumerate(
            zip(
                reversed_block_out_channels,
                reversed_num_layers_per_block,
                reversed_transformer_layers_per_block
            )
        ):
            prev_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i + 1, len(config.unet_block_out_channels) - 1)]
            if transformer_layers == 1:
                block = AttnUpBlock2D(
                    in_channels=in_channels,
                    prev_output_channel=prev_output_channel,
                    out_channels=out_channels,
                    temb_channels=config.unet_time_emb_dim,
                    num_layers=num_layers + 1,
                    resnet_groups=config.unet_resnet_groups,
                    attention_head_dim=out_channels // config.unet_num_heads,
                    upsample_type='conv' if (i < len(config.unet_block_out_channels) - 1) else None,
                )
            else:
                block = UpBlock2D(
                    in_channels=in_channels,
                    prev_output_channel=prev_output_channel,
                    out_channels=out_channels,
                    temb_channels=config.unet_time_emb_dim,
                    num_layers=num_layers + 1,
                    resnet_groups=config.unet_resnet_groups,
                    add_upsample=(i < len(config.unet_block_out_channels) - 1),
                )

            self.up_blocks.append(block)
            in_channels = out_channels

    def forward(
        self, hidden_states, time_steps, down_block_res_samples
    ):
        temb = self.time_embedding(time_steps.reshape(-1, 1))
        hidden_states = self.conv_in(hidden_states)
        
        for i, block in enumerate(self.up_blocks):            
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(block.resnets)]
            upsample_size = down_block_res_samples[-1].shape[2:] if not is_final_block else None
            hidden_states = block(hidden_states=hidden_states, res_hidden_states_tuple=res_samples, temb=temb, upsample_size=upsample_size)
        
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class SkipCausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        rope=None,
        text_rope=None,
        qk_norm=True,
        proj_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        clear_clean=False
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv_list = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(3)])
        self.q_norm_list = nn.ModuleList([norm_layer(self.head_dim) if qk_norm else nn.Identity() for _ in range(3)])
        self.k_norm_list = nn.ModuleList([norm_layer(self.head_dim) if qk_norm else nn.Identity() for _ in range(3)])
        self.proj_list = nn.ModuleList([nn.Linear(dim, dim, bias=proj_bias) for _ in range(3)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.caching, self.cached_k, self.cached_v = False, None, None
        self.rope = rope
        self.text_rope = text_rope
        self.clear_clean = clear_clean

    def set_caching(self, flag):
        self.caching, self.cached_k, self.cached_v = flag, None, None
        self.cache_buffer_k, self.cache_buffer_v = None, None
        
    def update_cache(self):
        if self.caching and self.clear_clean:
            self.cached_k = self.cache_buffer_k
            self.cached_v = self.cache_buffer_v
        

    def forward(self, x, position_ids=None, text_position_ids=None, attention_mask=None, block_size=None, cache=False, type_mask=None):
        B, N, C = x.shape
        
        text_x = x[:, type_mask == 0, :]
        clean_x = x[:, type_mask == 1, :]
        noised_x = x[:, type_mask == 2, :]
        
        text_qkv = self.qkv_list[0](text_x)
        text_qkv = text_qkv.reshape(B, text_qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        text_q, text_k, text_v = text_qkv.unbind(0)
        text_q, text_k = self.q_norm_list[0](text_q), self.k_norm_list[0](text_k)
        
        clean_qkv = self.qkv_list[1](clean_x)
        clean_qkv = clean_qkv.reshape(B, clean_qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        clean_q, clean_k, clean_v = clean_qkv.unbind(0)
        clean_q, clean_k = self.q_norm_list[1](clean_q), self.k_norm_list[1](clean_k)
        
        noised_qkv = self.qkv_list[2](noised_x)
        noised_qkv = noised_qkv.reshape(B, noised_qkv.size(1), 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        noised_q, noised_k, noised_v = noised_qkv.unbind(0)
        noised_q, noised_k = self.q_norm_list[2](noised_q), self.k_norm_list[2](noised_k)

        img_q = torch.cat((clean_q, noised_q), dim=2)
        img_k = torch.cat((clean_k, noised_k), dim=2)         
        if self.rope is not None:
            img_q, img_k = self.rope(img_q, img_k, position_ids)
        text_q, text_k = self.text_rope(text_q, text_k, text_position_ids)
        
        q = torch.cat((text_q, img_q), dim=2)
        k = torch.cat((text_k, img_k), dim=2)
        v = torch.cat((text_v, clean_v, noised_v), dim=2)
        
        if self.caching:
            if self.clear_clean:
                if self.cached_k is not None:
                    k = torch.cat((self.cached_k, k), dim=2)
                    v = torch.cat((self.cached_v, v), dim=2)
                if cache:
                    self.cache_buffer_k = k
                    self.cache_buffer_v = v
            else:
                if cache:
                    if self.cached_k is None:
                        self.cached_k = k[:, :, :-block_size, :]
                        self.cached_v = v[:, :, :-block_size, :]
                    else:
                        self.cached_k = torch.cat((self.cached_k, k[:, :, :block_size, :]), dim=2)
                        self.cached_v = torch.cat((self.cached_v, v[:, :, :block_size, :]), dim=2)

                if self.cached_k is not None:
                    k = torch.cat((self.cached_k, k[:, :, -block_size:, :]), dim=2)
                    v = torch.cat((self.cached_v, v[:, :, -block_size:, :]), dim=2)

        if not USE_FLEX_ATTENTION or not isinstance(attention_mask, BlockMask):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_drop.p
            )
        else:
            x = flex_attention(q, k, v, block_mask=attention_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = torch.cat((self.proj_list[0](x[:, type_mask==0, :]), self.proj_list[1](x[:, type_mask==1, :]), self.proj_list[2](x[:, type_mask==2, :])), dim=1)
        x = self.proj_drop(x)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        orig_dtype = q.dtype
        q = q.to(torch.float)
        k = k.to(torch.float)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

    def forward(self, q, k, position_ids):
        if position_ids is None or position_ids.shape[1] == 0:
            return q, k
        pos_ids = position_ids.float()
        theta = pos_ids.unsqueeze(-1) * self.inv_freq  # shape: (batch, seq_len, dim/2)
        cos = theta.cos()  # shape: (batch, seq_len, dim/2)
        sin = theta.sin()  # shape: (batch, seq_len, dim/2)
        cos = torch.cat([cos, cos], dim=-1)  # shape: (batch, seq_len, dim)
        sin = torch.cat([sin, sin], dim=-1)  # shape: (batch, seq_len, dim)
        q_embed, k_embed = self.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class MADFormerDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, rope=None, text_rope=None, qk_norm=True, clear_clean=False, **block_kwargs):
        super().__init__()
        self.self_attn = SkipCausalAttention(config.hidden_size, num_heads=config.num_attention_heads, qkv_bias=True, norm_layer=LlamaRMSNorm, qk_norm=qk_norm, rope=rope, text_rope=text_rope, clear_clean=clear_clean, **block_kwargs)
        self.mlp_list = nn.ModuleList([LlamaMLP(config) for _ in range(3)])
        self.input_layernorm_list = nn.ModuleList([LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])
        self.post_attention_layernorm_list = nn.ModuleList([LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])

    def forward(self, hidden_states, block_size, cache=False, attention_mask=None, position_ids=None, text_position_ids=None, type_mask=None, **kwargs):
        residual = hidden_states
        
        text_states = hidden_states[:, type_mask==0, :]
        clean_states = hidden_states[:, type_mask==1, :]
        noised_states = hidden_states[:, type_mask==2, :]
        hidden_states = torch.cat((self.input_layernorm_list[0](text_states), self.input_layernorm_list[1](clean_states), self.input_layernorm_list[2](noised_states)), dim=1)

        hidden_states = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask, 
            block_size=block_size, 
            cache=cache, 
            position_ids=position_ids,
            text_position_ids=text_position_ids,
            type_mask=type_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        text_states = self.post_attention_layernorm_list[0](hidden_states[:, type_mask==0, :])
        clean_states = self.post_attention_layernorm_list[1](hidden_states[:, type_mask==1, :])
        noised_states = self.post_attention_layernorm_list[2](hidden_states[:, type_mask==2, :])
        hidden_states = torch.cat((self.mlp_list[0](text_states), self.mlp_list[1](clean_states), self.mlp_list[2](noised_states)), dim=1)
        hidden_states = residual + hidden_states

        return hidden_states

    def set_caching(self, flag):
        self.self_attn.set_caching(flag)
        
    def update_cache(self):
        self.self_attn.update_cache()

def skip_causal_attn_mask_mod_gen(b, h, q_idx, kv_idx, block_size, len1, text_len):
    mask = torch.where(
        ((kv_idx < text_len) & (q_idx >= kv_idx)) |
        ((q_idx >= text_len) & (kv_idx >= text_len) & (((q_idx - text_len) // block_size) < len1) & (((kv_idx - text_len) // block_size) < len1) & (((q_idx - text_len) // block_size) >= ((kv_idx - text_len) // block_size))) | 
        ((q_idx >= text_len) & (kv_idx >= text_len) & (((q_idx - text_len) // block_size) >= len1) & (((kv_idx - text_len) // block_size) < len1) & ((((q_idx - text_len) // block_size) - len1) > ((kv_idx - text_len) // block_size))) | 
        ((q_idx >= text_len) & (kv_idx >= text_len) & (((q_idx - text_len) // block_size) == ((kv_idx - text_len) // block_size))), 
        True, 
        False
    )
    return mask

def skip_causal_attn_mask_mod_gen_no_clean(b, h, q_idx, kv_idx, block_size, len1, text_len):
    mask = torch.where(
        ((kv_idx < text_len) & (q_idx >= kv_idx)) |
        ((q_idx >= text_len) & (kv_idx >= text_len) & (((q_idx - text_len) // block_size) >= ((kv_idx - text_len) // block_size))), 
        True, 
        False
    )
    return mask

def blockwise_self_attn_mask_mod_gen(b, h, q_idx, kv_idx, block_size, len1, text_len):
    mask = torch.where(
        ((kv_idx < text_len) & (q_idx >= kv_idx)) |
        ((q_idx >= text_len) & (kv_idx >= text_len) & (((q_idx - text_len) // block_size) == ((kv_idx - text_len) // block_size))), 
        True, 
        False
    )
    return mask

class MADFormer(nn.Module):
    def __init__(
        self,
        patch_size=2,
        latent_size=16,
        block_size=256,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        diff_depth=14,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
        no_qk_norm=False,
        ar_len=4,
        spatial_len=1024,
        square_block=True,
        model_config_path=None,
        clear_clean=False,
        clear_cond=False,
        denoising_mlp=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.diff_depth = diff_depth
        self.head_dim = hidden_size // num_heads

        self.latent_size = latent_size
        self.block_size = block_size
        self.qk_norm = not no_qk_norm
        self.ar_len = ar_len
        self.spatial_len = spatial_len
        self.square_block = square_block
        self.clear_clean = clear_clean
        self.clear_cond = clear_cond
        self.denoising_mlp = denoising_mlp
        assert self.spatial_len == self.block_size * self.ar_len, f"Split block invalid: {self.spatial_len} != {self.block_size} * {self.ar_len}"

        self.model_config = MADFormerConfig(**json.load(open(model_config_path, "r")))
        self.model_config.hidden_size = hidden_size
        self.model_config.num_attention_heads = num_heads
       
        self.boi_block = nn.Parameter(torch.randn(1, self.block_size, self.hidden_size))
        self.downsampler = UNetDownsampler(self.model_config)
        self.upsampler = UNetUpsampler(self.model_config)
        
        h = w = math.isqrt(self.spatial_len) 
        self.rope = RopeND(nd=2, nd_split=[1, 1], max_lens=[h, w])
        
        self.max_len = self.model_config.max_position_embeddings
        self.text_rope = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=self.max_len, base=self.model_config.rope_theta)
        self.embed_tokens = nn.Embedding(self.model_config.vocab_size, hidden_size, self.model_config.pad_token_id)
        self.text_position_ids = torch.arange(self.max_len).unsqueeze(0)
        self.lm_head = nn.Linear(hidden_size, self.model_config.vocab_size, bias=False)
        
        def create_index_tensor(max_lens):
            ranges = [torch.arange(m) for m in max_lens]
            grids = torch.meshgrid(*ranges, indexing='ij')
            return torch.stack(grids).reshape(len(max_lens), -1)
        self.position_ids_precompute = create_index_tensor([h, w])
        if square_block:
            position_idx = torch.arange(self.spatial_len)
            h = w = int(math.isqrt(self.spatial_len))
            position_idx = position_idx.view(h, w)
            block_h = block_w = int(math.isqrt(self.block_size))
            position_idx = einops.rearrange(position_idx, '(new_h block_h) (new_w block_w)-> new_h new_w block_h block_w', block_h=block_h, block_w=block_w)
            position_idx = einops.rearrange(position_idx, 'new_h new_w block_h block_w -> (new_h new_w) (block_h block_w)')
            position_idx = position_idx.view(-1)
            self.position_ids_precompute = self.position_ids_precompute[:, position_idx]
        self.blocks = nn.ModuleList([
            MADFormerDecoderLayer(self.model_config, layer_idx=i, rope=self.rope, text_rope=self.text_rope, qk_norm=self.qk_norm, clear_clean=self.clear_clean) for i in range(self.depth)
        ])
        self.initialize_weights()
        self.init_flex_attn()

    def init_flex_attn(self):
        if USE_FLEX_ATTENTION:
            blockwise_self_attn_mask_mod = partial(blockwise_self_attn_mask_mod_gen, block_size=self.block_size, len1=self.ar_len, text_len=self.max_len)
            if self.clear_clean:
                skip_causal_attn_mask_mod = partial(skip_causal_attn_mask_mod_gen_no_clean, block_size=self.block_size, len1=self.ar_len, text_len=self.max_len)
                self.flex_attnmask = create_block_mask(skip_causal_attn_mask_mod, B=None, H=None, Q_LEN=self.max_len+self.ar_len*self.block_size, KV_LEN=self.max_len+self.ar_len*self.block_size)
                self.flex_attnmask_mlp = create_block_mask(blockwise_self_attn_mask_mod, B=None, H=None, Q_LEN=self.max_len+self.ar_len*self.block_size, KV_LEN=self.max_len+self.ar_len*self.block_size)
            else:
                skip_causal_attn_mask_mod = partial(skip_causal_attn_mask_mod_gen, block_size=self.block_size, len1=self.ar_len, text_len=self.max_len)
                self.flex_attnmask = create_block_mask(skip_causal_attn_mask_mod, B=None, H=None, Q_LEN=self.max_len+2*self.ar_len*self.block_size, KV_LEN=self.max_len+2*self.ar_len*self.block_size)
                self.flex_attnmask_mlp = create_block_mask(blockwise_self_attn_mask_mod, B=None, H=None, Q_LEN=self.max_len+2*self.ar_len*self.block_size, KV_LEN=self.max_len+2*self.ar_len*self.block_size)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def build_attention_mask(self, T, B, device):
        size = T * B
        m_noise_noise = torch.zeros(size, size)
        for i in range(T):
            start_idx = i * B
            end_idx = start_idx + B
            m_noise_noise[start_idx:end_idx, start_idx:end_idx] = torch.ones(B, B)
        m_noise_clean = torch.zeros(size, size)
        for i in range(T):
            for j in range(i + 1, T):
                start_col = i * B
                end_col = start_col + B
                start_row = j * B
                end_row = start_row + B
                m_noise_clean[start_row:end_row, start_col:end_col] = 1
        m_clean_noise = torch.zeros(size, size)
        m_clean_clean = torch.zeros(size, size)
        for i in range(T):
            start_idx = i * B
            end_idx = start_idx + B
            m_clean_clean[start_idx:end_idx, :end_idx] = 1

        attn_mask = torch.zeros(2 * size, 2 * size)
        attn_mask[:size, :size] = m_clean_clean
        attn_mask[:size, size:] = m_clean_noise
        attn_mask[size:, :size] = m_noise_clean
        attn_mask[size:, size:] = m_noise_noise
        return attn_mask.bool().to(device) if not self.clear_clean else m_clean_clean.bool().to(device)

    def build_inference_attention_mask(self, block_id, text_len, B, device):
        attention_mask = torch.ones(2 * B, text_len + B * (block_id + 1))
        attention_mask[:B, -B:] = torch.zeros(B, B)
        attention_mask = attention_mask.bool().to(device)
        return attention_mask

    def build_block_self_attn_inference_attention_mask(self, block_id, text_len, B, device):
        if self.clear_clean:
            attention_mask = torch.zeros(B, text_len + B * (block_id + 1))
            attention_mask[:, -B:] = torch.ones(B, B)
        else:
            attention_mask = torch.zeros(2 * B, text_len + B * (block_id + 1))
            attention_mask[:B, -2*B:-B] = torch.ones(B, B)
            attention_mask[B:, -B:] = torch.ones(B, B)
        attention_mask = attention_mask.bool().to(device)
        return attention_mask

    def forward(self, input_ids, clean_x, noised_x, t):
        N, T, B, _ = clean_x.size()    
        hidden_states_text = self.embed_tokens(input_ids)
        text_position_ids = self.text_position_ids.to(clean_x.device).expand(N, -1)
        
        clean_x = einops.rearrange(clean_x, 'N T (H W) C -> (N T) C H W', H=int(math.isqrt(B)))
        noised_x = einops.rearrange(noised_x, 'N T (H W) C -> (N T) C H W', H=int(math.isqrt(B)))
        clean_x, clean_x_res = self.downsampler(clean_x, t)
        noised_x, noised_x_res = self.downsampler(noised_x, t)
        clean_x = einops.rearrange(clean_x, '(N T) C H W -> N (T H W) C', T=T)
        noised_x = einops.rearrange(noised_x, '(N T) C H W -> N (T H W) C', T=T)

        shifted_clean_x = torch.cat((self.boi_block.expand(N, -1, -1), clean_x[:, :-self.block_size, :]), dim=1)
        if self.clear_clean:
            position_ids = self.position_ids_precompute
            x = shifted_clean_x
            type_mask = torch.full_like(shifted_clean_x, 2)
        else:
            position_ids=torch.cat([self.position_ids_precompute , self.position_ids_precompute], dim=-1) # for ACDiT clean and noise
            x = torch.cat((clean_x, shifted_clean_x), dim=1)
            type_mask = torch.cat((torch.ones_like(clean_x), torch.full_like(shifted_clean_x, 2)), dim=1)
        
        x = torch.cat((hidden_states_text, x), dim=1)
        type_mask = torch.cat((torch.zeros_like(hidden_states_text), type_mask), dim=1)[0, :, 0]
        
        if not USE_FLEX_ATTENTION:
            raise NotImplementedError("Normal attn mask for denoising_mlp=True and text not implemented!")
            attention_mask = self.build_attention_mask(T, B, x.device)
        else:
            attention_mask = self.flex_attnmask
            attention_mask_mlp = self.flex_attnmask_mlp if self.denoising_mlp else self.flex_attnmask

        for i, block in enumerate(self.blocks):
            block_attn_mask = attention_mask if i < self.depth - self.diff_depth else attention_mask_mlp
            if i == self.depth - self.diff_depth:
                condition = x[:, type_mask==2, :]
                if self.clear_cond:
                    x[:, type_mask==2, :] = noised_x
                else:
                    x[:, type_mask==2, :] += noised_x
            x = block(x, self.block_size, cache=False, attention_mask=block_attn_mask, 
                      position_ids=position_ids, text_position_ids=text_position_ids, type_mask=type_mask)
            
        logits = self.lm_head(x[:, type_mask==0, :])
        if self.clear_clean:
            clean_tower_output = None
        else:
            clean_tower_output = einops.rearrange(x[:, type_mask==1, :], 'N (T H W) C -> (N T) C H W', T=T, H=int(math.isqrt(self.block_size)))
            clean_tower_output = self.upsampler(clean_tower_output, t, clean_x_res)
        condition = einops.rearrange(condition, 'N (T H W) C -> (N T) C H W', T=T, H=int(math.isqrt(self.block_size)))
        condition = self.upsampler(condition, t, clean_x_res)
        x = einops.rearrange(x[:, type_mask==2, :], 'N (T H W) C -> (N T) C H W', T=T, H=int(math.isqrt(self.block_size)))
        x = self.upsampler(x, t, noised_x_res)
        return x, condition, clean_tower_output, logits

    @torch.no_grad()
    def sample(self, input_ids, scheduler, num_inference_steps, target_shape, generator=None, dtype=torch.bfloat16):
        N, T, B, C = target_shape
        hidden_states_text = self.embed_tokens(input_ids)
        text_position_ids = self.text_position_ids.to(input_ids.device).expand(N, -1)
        scheduler.set_timesteps(num_inference_steps, device='cuda')

        clean_latents = []
        for block in self.blocks:
            block.set_caching(True)
            
        for block_id in range(T):
            # 0. Calculate pos ids for both passes
            noise_ids = self.position_ids_precompute[:, block_id*self.block_size:(block_id+1)*self.block_size]
            if block_id > 0 and not self.clear_clean:
                clean_ids = self.position_ids_precompute[:, (block_id-1)*self.block_size:block_id*self.block_size]
                position_ids = torch.cat([clean_ids, noise_ids], dim=-1) # for AC DiT clean and noise
            else:
                position_ids = noise_ids

            # 1. AR pass to generate condition
            # NOTE: Caching logic here critical for clear_clean=True
            cache_flag = block_id < T-1 if self.clear_clean else True
            if block_id > 0:
                clean_x = einops.rearrange(clean_latents[-1], 'N T (H W) C -> (N T) C H W', H=int(math.isqrt(B)))
                clean_x, clean_x_res = self.downsampler(clean_x, t)
                clean_x = einops.rearrange(clean_x, '(N T) C H W -> N (T H W) C', T=1)
                if not self.clear_clean:
                    x = torch.cat((clean_x, clean_x), dim=1)
                    attention_mask = self.build_inference_attention_mask(block_id, self.max_len, self.block_size, x.device)
                    type_mask = torch.cat((torch.ones_like(clean_x), torch.full_like(clean_x, 2)), dim=1)[0, :, 0]
                else:
                    x = clean_x
                    attention_mask = None
                    type_mask = torch.full_like(clean_x, 2)[0, :, 0]
            else:
                x = torch.cat((hidden_states_text, self.boi_block.expand(N, -1, -1)), dim=1)
                attention_mask = torch.tril(torch.ones(x.size(1), x.size(1)))
                attention_mask[-self.block_size:, -self.block_size:] = 1 
                attention_mask = attention_mask.bool().to(x.device)
                type_mask = torch.cat((torch.zeros_like(hidden_states_text),  torch.full_like(self.boi_block.expand(N, -1, -1), 2)), dim=1)[0, :, 0]
            for i, block in enumerate(self.blocks[:-self.diff_depth]):
                x = block(x, self.block_size, cache=cache_flag, attention_mask=attention_mask, position_ids=position_ids, text_position_ids=text_position_ids, type_mask=type_mask)
            
            condition = x
            if self.clear_cond:
                condition[:, type_mask==2, :] = 0     

            # 2. Denoising passes
            noised_x = randn_tensor((N, 1, B, C), device='cuda', generator=generator, dtype=dtype)
            scheduler.set_timesteps(num_inference_steps, device='cuda')
            if self.denoising_mlp:
                attention_mask = self.build_block_self_attn_inference_attention_mask(block_id, self.max_len, self.block_size, x.device) if block_id > 0 else None
            for t in scheduler.timesteps:
                # NOTE: Caching logic here flexible for clear_clean=True
                cache_flag = (block_id < T-1) and (t == scheduler.timesteps[-1]) if self.clear_clean \
                    else (t == scheduler.timesteps[0])
                timesteps = torch.tensor([t] * noised_x.size(0), device='cuda')
                
                noised_x = scheduler.scale_model_input(noised_x, t)
                noised_x_square = einops.rearrange(noised_x, 'N T (H W) C -> (N T) C H W', H=int(math.isqrt(B)))
                noised_x_downsampled, noised_x_res = self.downsampler(noised_x_square, timesteps)
                noised_x_downsampled_square = einops.rearrange(noised_x_downsampled, '(N T) C H W -> N (T H W) C', T=1)
                
                x = condition.clone()
                x[:, type_mask==2, :] += noised_x_downsampled_square
                for i, block in enumerate(self.blocks[-self.diff_depth:]):
                    x = block(x, self.block_size, cache=cache_flag, attention_mask=attention_mask, position_ids=position_ids, text_position_ids=text_position_ids, type_mask=type_mask)

                x = einops.rearrange(x[:, type_mask==2, :], 'N (H W) C -> N C H W', H=int(math.isqrt(self.block_size)))
                noise_pred = self.upsampler(x, timesteps, noised_x_res)
                noise_pred = einops.rearrange(noise_pred, 'N C H W -> N (H W) C').unsqueeze(1)
                noised_x = scheduler.step(noise_pred, t, noised_x).prev_sample
                
            for block in self.blocks:
                block.update_cache()
            clean_latents.append(noised_x)
            text_position_ids = None
            
        clean_latents = torch.cat(clean_latents, dim=1)
        for block in self.blocks:
            block.set_caching(False)
            
        return clean_latents
