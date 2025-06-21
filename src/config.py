import json
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import List, Optional, Tuple, Union, Dict

class MADFormerConfig(LlamaConfig):
    def __init__(self, 
                 vae_config_path: str = "",
                 vae_path: str = "stabilityai/sd-vae-ft-mse",
                 unet_block_out_channels: List[int] = [512, 1024],
                 unet_num_layers_per_block: List[int] = [2, 2],
                 unet_resnet_groups: int = 8,
                 unet_transformer_layers_per_block: List[int] = [0, 1],
                 unet_num_heads: int = 8,
                 unet_time_emb_dim: int = 128,
                 **kwargs
                ):
        '''
        Params:
            vae_path: path to vae
            vae_config_path: local path to vae's config

            unet_block_out_channels (list): Number of output channels per block (reverse order of downsampler).
            unet_num_layers_per_block (list): Number of ResNet layers in each block.
            unet_resnet_groups (int): Number of groups in ResNet layers.
            unet_transformer_layers_per_block (list): Number of transformer layers per block.
            unet_num_heads (int): Number of attention heads for transformer blocks.
            unet_time_emb_dim (int): Dim of time embedding passed into ResNet blocks.                    
        '''
        super().__init__(**kwargs)
        self.vae_path = vae_path
        self.vae_config = self.get_vae_config(vae_config_path)
        self.latent_patch_dim = self.vae_config["latent_channels"]
        self.unet_block_out_channels = unet_block_out_channels
        self.unet_num_layers_per_block = unet_num_layers_per_block
        self.unet_resnet_groups = unet_resnet_groups
        self.unet_transformer_layers_per_block = unet_transformer_layers_per_block
        self.unet_num_heads = unet_num_heads
        self.unet_time_emb_dim = unet_time_emb_dim
        
    def get_vae_config(self, vae_config_path: str):
        with open(vae_config_path, 'r') as f:
            vae_config = json.load(f)
        return vae_config
