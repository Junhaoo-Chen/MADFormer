<h1 align="center">MADFormer: Mixed Autoregressive and Diffusion Transformers for Continuous Image Generation</h1>

This repository contains the official implementation of **MADFormer**, a unified generative model that fuses the global modeling of **autoregressive transformers** with the fine-grained refinement capabilities of **diffusion models**. MADFormer introduces a flexible, two-axis hybrid framework—mixing AR and diffusion across spatial blocks and model layers—delivering strong performance under compute constraints while maintaining high visual fidelity across image generation tasks.

<div align="center">
<a href='https://arxiv.org/pdf/2506.07999'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
</div>

## Overview

**MADFormer** (**M**ixed **A**utoregressive and **D**iffusion Transformer) bridges the strengths of autoregressive (AR) and diffusion-based generation through a unified architecture designed for **continuous image synthesis**. It introduces a **dual-axis hybridization strategy**, mixing AR and diffusion both across **image blocks** (token axis) and **transformer layers** (depth axis). This enables:

- **Global context modeling** through autoregressive conditioning across image blocks.
- **Local detail refinement** via diffusion modeling within each block’s continuous latent space.
- **Flexible capacity allocation**, allowing principled trade-offs between speed and quality.

> TODO: Add images for model arch

The generation process of MADFormer: each image block is autoregressively predicted, then refined through a conditioned diffusion process.  

MADFormer acts not only as a performant generator for high-resolution data like **FFHQ-1024** and regular images like **ImageNet-256**, but also as a **testbed** for exploring hybrid design choices. Notably, we show that **increasing AR layer allocation** can improve FID by up to **60–75%** under constrained inference budgets. Our modular design supports controlled experiments on inference cost, block granularity, loss objectives, and layer allocation—offering actionable insights for hybrid model design in multimodal generation.

## Models
We provide the model weights for MADFormer trained on FFHQ-1024 and ImageNet through the download links below.

| Dataset          | 🤗 Link          |
|:----------------:|:----------------:|
| **FFHQ-1024** | [MADFormer-FFHQ](https://huggingface.co/JunhaoC/MADFormer-FFHQ/blob/main/ckpts.pt) |
| **ImageNet**  | [MADFormer-ImageNet](https://huggingface.co/JunhaoC/MADFormer-ImageNet/blob/main/ckpts.pt) |

## Setup
To set up the runtime environment for this project, install the required dependencies using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

## Training
Our training configurations are provided in the `configs` directory, complete with model and training hyperparameters. You can use the following command to start training:
```bash
cd src
python train.py <training_args>
```

## Sampling
After downloading the checkpoints or finishing pretraining, you can use the following scripts to generate images:
```bash
cd src
python sample.py --ckpt <path_to_checkpoint> --sample_steps <number_of_steps> --save_dir <output_directory> --range_start <id_of_first_image> --range_end <id_of_last_image> --max_bs <max_batch_size>
```

## Evaluation

We adopt **Fréchet Inception Distance (FID)** as our primary evaluation metric for image quality. For **FFHQ-1024**, FID is computed over 8,000 generated samples. Image generation is performed with the **DDIM sampler** , using 250 sampling steps for FFHQ. To ensure stability, final FID scores are averaged across the last five checkpoints (saved every 10,000 steps). 

FID scores are computed using the [pytorch-fid](https://pypi.org/project/pytorch-fid/) library.

## Acknowledgements
This code is mainly built upon the [ACDiT](https://github.com/thunlp/ACDiT) repository.

## License
This project is liscenced under the Apache-2.0 liscence.

## Citation

If you find MADFormer useful in your research, please consider citing our paper:

```bibtex
@article{MADFormer,
    title={MADFormer: Mixed Autoregressive and Diffusion Transformers for Continuous Image Generation}, 
    author={Junhao Chen and Yulia Tsvetkov and Xiaochuang Han},
    journal={arXiv preprint arXiv:2506.07999},
    year={2025}
}
```