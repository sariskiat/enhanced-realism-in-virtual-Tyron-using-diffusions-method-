import torch
import numpy as np
from copy import deepcopy
import PIL
import PIL
from copy import deepcopy
def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask
def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = deepcopy(mask)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, i - 1, j] = 0
                if j != 0:
                    mask[:, i, j - 1] = 0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        prepared_masks.append(mask)
    return torch.stack(prepared_masks, dim=0)


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask_and_masked_image
def prepare_mask_and_masked_image(image, mask, height, width):
    r"""
    Prepares a pair (mask, image) to be consumed by the Kandinsky inpaint pipeline. This means that those inputs will
    be converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for
    the ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    mask = 1 - mask

    return mask, image

import PIL
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

clean_mem()
import logging
num_images_per_prompt = 1
logging.disable(logging.WARNING)

set_seed(42)
# if fc.defaults.cpus > 8: 
fc.defaults.cpus = 0
from accelerate import Accelerator

acc = Accelerator()
device = acc.device
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.utils.torch_utils import randn_tensor
# pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
# ).to(device)
# clip = pipe_prior.image_encoder.to(device).to(torch.float16).requires_grad_(False)

# tokz = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32)
# txt_enc = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32).to(
#     device).requires_grad_(False)

# prompt = ['Clothes']
# w, h = 256, 256
# n_inf_steps = 77
# g_scale = 7.5
# bs = 200
# seed = 77
# txt_inp = tokz(
#     prompt * bs,
#     padding='max_length',
#     max_length=tokz.model_max_length,
#     truncation=True,
#     return_tensors='pt'
# )
do_classifier_free_guidance = 4.0 > 1.0
# txt_emb = txt_enc(txt_inp['input_ids'].to(device))[0]
from transformers import CLIPVisionModel,CLIPTextModel,CLIPVisionModelWithProjection


def prepare_image(pil_image, w=32, h=32):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float16) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image

def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor
def abar(t): return (t * math.pi / 2).cos() ** 2


def inv_abar(x): return x.sqrt().acos() * 2 / math.pi

# def noisify(latents):
#     device = latents.device
#     n = len(latents)
#     # t = torch.rand(n, ).to(x0).clamp(0, 0.999)
#     t = np.random.triangular(0, 0.5, 1, (n,)) * 1000
#     t = torch.tensor(t, dtype=torch.float16, device=device)
#     # t = torch.randint(0, 1000, device=device, dtype=torch.long)?
#     ε = torch.randn(latents.shape, device=device)
#     abar_t = abar(t).reshape(-1, 1, 1, 1).to(device)
#     xt = abar_t.sqrt() * latents + (1 - abar_t).sqrt() * ε
#     return (xt.to(torch.float16), t.to(torch.float16).to(device)), ε.to(torch.float16)

def noisify(latents):
    
    noise = torch.randn(latents.shape, device=latents.device)
    bs = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (bs,), device=latents.device,
        dtype=torch.int64
    )
    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_images = scheduler.add_noise(latents, noise, timesteps)
    # latent_model_input = torch.cat([noisy_image] * 2) if do_classifier_free_guidance else noisy_image
    # noise = torch.cat([noise] * 2) if do_classifier_free_guidance else noise



    return (noisy_images.to(torch.float16), timesteps.to(torch.float16).to(device)), noise.to(torch.float16)
def init_ddpm(model):
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers): init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()
    
        




def custom_collate_fn(batch):
    # Stack images along the batch dimension
    return torch.stack(batch, dim=0)


def flat_mse(x, y): 

    return F.mse_loss(x.flatten(), y.flatten())