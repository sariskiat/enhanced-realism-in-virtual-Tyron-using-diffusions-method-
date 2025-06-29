import sys

sys.path.append('/Users/saris/Desktop/thesiswork/hr-vton/StableVITON')
sys.path.append("/Users/saris/Desktop/thesiswork/hr-vton/HR-VITON/data/dataset.py")
sys.path.append("/Users/saris/miniconda3/envs/my_env/lib/python3.10/site-packages/miniai")
sys.path.append("/Users/saris/Downloads/models.py")
from models import load_unet, load_prior_pipeline, load_controlnet_pipeline , load_emb
import torch
from diffusers import UNet2DConditionModel, KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline
import timm, torch, random, datasets, math, fastcore.all as fc, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import k_diffusion as K, torchvision.transforms as T
import torchvision.transforms.functional as TF, torch.nn.functional as F

from torch.utils.data import DataLoader, default_collate
from pathlib import Path
from torch.nn import init
from fastcore.foundation import L
from torch import nn, tensor
from datasets import load_dataset
from operator import itemgetter
from torcheval.metrics import MulticlassAccuracy
from functools import partial
from torch.optim import lr_scheduler
from torch import optim

from miniai.datasets import *
from miniai.conv import *
from miniai.learner import *
from miniai.activations import *
from miniai.init import *
from miniai.sgd import *
from miniai.resnet import *
from miniai.augment import *
from miniai.accel import *
from fastprogress import progress_bar
from diffusers import UNet2DModel, DDIMPipeline, DDPMPipeline, DDIMScheduler, DDPMScheduler

torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray_r'
mpl.rcParams['figure.dpi'] = 70
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle, gzip, math, os, time, shutil, torch, random, logging
import fastcore.all as fc, matplotlib as mpl, numpy as np, matplotlib.pyplot as plt
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter, itemgetter
from functools import partial
from copy import copy
from contextlib import contextmanager

from fastcore.foundation import L
import torchvision.transforms.functional as TF, torch.nn.functional as F
from torch import tensor, nn, optim
from torch.utils.data import DataLoader, default_collate
from torch.nn import init
from torch.optim import lr_scheduler
from torcheval.metrics import MulticlassAccuracy
from datasets import load_dataset, load_dataset_builder

from miniai.datasets import *
from miniai.conv import *
from miniai.learner import *
from miniai.activations import *
from miniai.init import *
from miniai.sgd import *
from miniai.resnet import *
from miniai.augment import *
from miniai.imports import *
from miniai.diffusion import *
from models import load_unet, load_prior_pipeline, load_controlnet_pipeline
from glob import glob
from fastprogress import progress_bar
from diffusers import AutoencoderKL, UNet2DConditionModel
import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.io import read_image, ImageReadMode
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image, ImageReadMode
from glob import glob
from pathlib import Path
from tqdm import tqdm


import logging


# if fc.defaults.cpus > 8: 
fc.defaults.cpus = 0
from accelerate import Accelerator

acc = Accelerator()
device = acc.device
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
    SimpleCrossAttnDownBlock2D,
    UNetMidBlock2DSimpleCrossAttn,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

import os

# Set the environment variable
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Example: DDPMCB callback (simplified, you may need to adjust imports and dependencies)

pipe_prior = load_prior_pipeline(device)
pipe_emb = load_emb(device)
clip = pipe_prior.image_encoder.to(device).to(torch.float16).requires_grad_(False)
pipe = load_controlnet_pipeline(device)
movq = pipe.movq.to(device).to(torch.float16)
scheduler = pipe.scheduler
num_inference_steps: int = 100
guidance_scale: float = 4.0
num_images_per_prompt: int = 1
negativepromt   = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
# Usage example
class DDPMCB(TrainCB):
    pipe_prior = load_prior_pipeline(device)
    pipe = load_controlnet_pipeline(device)

    clip = pipe_prior.image_encoder.to(device).to(torch.float16).requires_grad_(False)
    order = DeviceCB.order + 1
    model_cpu_offload_seq = "unet->movq"
    def __init__(self
        ):
        super().__init__()

        self.movq_scale_factor = 2 ** (len(movq.config.block_out_channels) - 1)
        self._warn_has_been_called = False
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma
        return latents

    def predict(self, learn):
        # for i in range(2):
            
        #     down_samples, mid_sample = multi[i](
        #             learn.batch[0][0],
        #             learn.batch[0][1].to(device),
        #             encoder_hidden_states=None,
        #             controlnet_cond=learn.batch[0][3][i],
        #             conditioning_scale=0.75,
        #             guess_mode=False,
        #             return_dict=False,
        #             added_cond_kwargs=learn.batch[0][2],
        #         )
    
        #     # merge samples
        #     if i == 0:
        #         down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        #     else:
        #         down_block_res_samples = [
        #             samples_prev + samples_curr
        #             for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
        #         ]
        #         mid_block_res_sample += mid_sample
        # down_block_res_samples, mid_block_res_sample = xx(
        #             learn.batch[0][0],
        #             learn.batch[0][1].to(device),
        #             encoder_hidden_states=None,
        #             controlnet_cond=learn.batch[0][3],
        #             conditioning_scale=0.75,
        #             guess_mode=False,
        #             return_dict=False,
        #             added_cond_kwargs=learn.batch[0][2],
        #         )
        learn.preds = learn.model(
                sample=learn.batch[0][0],
        timestep=learn.batch[0][1].to(device),
        encoder_hidden_states=None,
        added_cond_kwargs=learn.batch[0][2],
                return_dict=False,
                # down_block_additional_residuals=down_block_res_samples,
                # mid_block_additional_residual=mid_block_res_sample
            )[0]
        if 4>1:
                learn.preds, variance_pred = learn.preds.split(4, dim=1)
                noise_pred_uncond, noise_pred_text = learn.preds.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                learn.preds = noise_pred_uncond + 4 * (noise_pred_text - noise_pred_uncond)
                learn.preds = torch.cat([learn.preds, variance_pred_text], dim=1)
        learn.preds, _ = learn.preds.split(4, dim=1)
        return learn.preds

        # learn.preds = learn.model(*learn.batch[0]).sample
        
        # print("pred ",learn.preds.mean())

    def before_batch(self, learn):
        # Print the initial state of learn.batch
        batch = learn.batch
        batch = batch
        
        cloth = batch[2].permute(0,3,1,2).to(device).to(torch.float16)
        
        
        cloth = TF.resize(cloth,(224,224))
        cloth_cond = TF.resize(cloth,(768,768))
        open_pose = batch[4].permute(0,3,1,2).to(device).to(torch.float16)
        image= batch[0].permute(0,3,1,2).to(device).to(torch.float16)
        image = TF.resize(image,(768,768))
        # mask= batch[1].permute(0,3,1,2).to(device).to(torch.float16)
        # mask = TF.resize(mask,(768,768))
        # mask_image = mask.cpu().numpy()


        open_pose = batch[4].permute(0,3,1,2).to(device).to(torch.float16)
        open_pose = TF.resize(open_pose,(768,768))

        image_embeds = clip(cloth.to(device).to(torch.float16)).image_embeds.to(torch.float16)
        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        negative_emb = pipe_emb(prompt=negativepromt, image=cloth, strength=1).image_embeds
        negative_image_embeds = negative_emb
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)
        if 4>1:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
            dtype=torch.float16, device=device
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        # preprocess image and mask
        # mask_image = torch.from_numpy(mask_image)
        mask = np.zeros((768, 768), dtype=np.float16)

# Mask out an area above the cat's head
        mask[250:500, 200:-200] = 1
        mask_tensor = torch.from_numpy(mask)
        mask_image, _ = prepare_mask_and_masked_image(image, mask_tensor, 768, 768)

        # image = image.to(dtype=image_embeds.dtype, device=device)
        open_pose = open_pose.to(dtype=image_embeds.dtype, device=device)
        image = movq.encode(image)["latents"]
        open_pose_latents = movq.encode(open_pose)["latents"]
        cloth_cond_latents = movq.encode(cloth_cond)["latents"]
        cond_image = TF.resize(cloth,(768,768))

        mask_image = mask_image.to(dtype=image_embeds.dtype, device=device)
        image_shape = tuple(image.shape[-2:])
        mask_image = F.interpolate(
            mask_image,
            image_shape,
            mode="nearest",
        )
        mask_image = prepare_mask(mask_image)
        masked_image = image * mask_image
        open_pose_latents = open_pose_latents * mask_image


        mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
        masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)

        if 1==1:
            mask_image = mask_image.repeat(2, 1, 1, 1)
            masked_image = masked_image.repeat(2, 1, 1, 1)
            # cond_image = cond_image.repeat(2,1,1,1)
        # num_channels_latents = movq.config.latent_channels
        # height, width = downscale_height_and_width(768, 768, self.movq_scale_factor)
        # create initial latent
        latents = image
        (latents, t), ε = noisify(latents.to(torch.float16))
        cond_image = torch.cat([cond_image] * 2) if 4>1 else cond_image
        latent_model_input = torch.cat([latents] * 2) if 4>1 else latents
        open_pose_latents = torch.cat([open_pose_latents] * 2) if 4>1 else open_pose_latents
        cloth_cond_latents = torch.cat([cloth_cond_latents] * 2) if 4>1 else cloth_cond_latents

        latent_model_input = torch.cat([latent_model_input, masked_image, mask_image], dim=1)   
        open_pose = torch.cat([open_pose] * 2) if 4>1 else open_pose
        added_cond_kwargs = {"image_embeds": image_embeds.to(torch.float16)}
        conds = [cond_image.to(device),open_pose.to(device)] #temppause to use latenets to conditioned instead
        # conds = [cloth_cond_latents.to(device),open_pose_latents.to(device)]
        # print(xt.shape,mask.shape,mask_latent.shape)
        learn.batch = ((latent_model_input.to(device), t.to(device),added_cond_kwargs,conds
                ), ε.to(device))

    def after_pred(self, learn):
        # Print output shape
        print(f"Output shape (preds): {learn.preds.shape}")
        clean_mem()

    # def backward(self, learn):
    #     learn.loss.backward(retain_graph=True)  # Retain the graph for another backward pass if needed

    # @torch.no_grad()
    # def sample(self, f, model, sz, steps, eta=1.):
    #     ts = torch.linspace(1 - 1 / steps, 0, steps)
    #     x_t = torch.randn(sz).to(model.device)
    #     preds = []
    #     for i, t in enumerate(progress_bar(ts)):
    #         abar_t = abar(t)
    #         noise = model(x_t, t).sample
    #         abar_t1 = abar(t - 1 / steps) if t >= 1 / steps else torch.tensor(1, dtype=torch.float32)
    #         x_0_hat, x_t = f(x_t, noise, abar_t, abar_t1, 1 - abar_t, 1 - abar_t1, eta, 1 - ((i + 1) / 100))
    #         preds.append(x_0_hat.float().cpu().to(torch.float32))
    #     return preds
