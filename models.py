import sys

sys.path.append('/Users/saris/Desktop/thesiswork/hr-vton/StableVITON')
sys.path.append("/Users/saris/Desktop/thesiswork/hr-vton/HR-VITON/data/dataset.py")
sys.path.append("/Users/saris/miniconda3/envs/my_env/lib/python3.10/site-packages/miniai")
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
import torch
import numpy as np

from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from diffusers.utils import load_image
from transformers import pipeline


def load_unet(device):
    unet = UNet2DConditionModel.from_pretrained(
        'sariskiat/unet32',
        mid_block_type='UNetMidBlock2DCrossAttn',
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
    ).requires_grad_(True).to(device)
    return unet

def load_prior_pipeline(device):
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior', torch_dtype=torch.float16
    ).to(device)
    return pipe_prior
def load_emb(device):
    pipe_emb = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior', torch_dtype=torch.float16
    ).to(device)
    return pipe_emb

def load_controlnet_pipeline(device):
    pipe = KandinskyV22ControlnetPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-controlnet-depth', torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe
