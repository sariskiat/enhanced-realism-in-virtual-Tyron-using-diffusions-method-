import sys

sys.path.append('/Users/saris/Desktop/thesiswork/hr-vton/StableVITON')
sys.path.append("/Users/saris/Desktop/thesiswork/hr-vton/HR-VITON/data/dataset.py")
sys.path.append("/Users/saris/miniconda3/envs/my_env/lib/python3.10/site-packages/miniai")
sys.path.append("/Users/saris/Downloads/models.py")
from models import load_unet, load_prior_pipeline, load_controlnet_pipeline
from utils import flat_mse
from callbacks import DDPMCB
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
from importlib import import_module

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


def main():
    device = torch.device('mps')

    # Load datasets
    train_dataset = getattr(import_module("dataset"), "VITONHDDataset")(
        data_root_dir="/Users/saris/Desktop/thesiswork/hr-vton/HR-VITON/data",
        img_H=768,
        img_W=768,
    )
    valid_paired_dataset = getattr(import_module("dataset"), "VITONHDDataset")(
        data_root_dir="/Users/saris/Desktop/thesiswork/hr-vton/HR-VITON/data",
        img_H=768,
        img_W=768,
        is_test=True,
        is_paired=True,
        is_sorted=True,
    )

    def collate_ddpm(b):
        b = default_collate(b)
        image = b['image']
        mask = b['gt_cloth_warped_mask']
        cloth = b['cloth']
        control_image = b['agn']
        open_pose = b['image_densepose']
        return (image, mask, cloth, control_image, open_pose)

    from torch.utils.data import DataLoader
    from fastcore.foundation import L

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_ddpm
    )
    valid_paired_dataloader = DataLoader(
        valid_paired_dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_ddpm
    )

    # DataLoaders wrapper (from miniai)
 

    dl = DataLoaders(train_dataloader, valid_paired_dataloader)

    # Load models and pipelines
    unet = load_unet(device)


    # Callback
    from fastcore.foundation import L

    # Set up training
    lr = 1e-7
    epochs = 1
    opt_func = partial(optim.Adam, eps=1e-5)
    tmax = epochs * len(dl.train)
    sched = partial(lr_scheduler.OneCycleLR, max_lr=lr, total_steps=tmax)
    cbs = [DeviceCB(device=device), DDPMCB(), ProgressCB(plot=True), MetricsCB(), BatchSchedCB(sched)]

    learn = Learner(unet, dl, flat_mse, lr=lr, cbs=cbs, opt_func=opt_func)
    learn.fit(epochs)

if __name__ == "__main__":
    main()
