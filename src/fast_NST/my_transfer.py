#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import os
import argparse
import time


# In[2]:


from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# In[4]:


import utils
from network import ImageTransformNet, Vgg16
#from vgg import Vgg16


# In[5]:


# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7


# In[6]:


def enable_gpu(args):
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())
        return use_cuda



def style_transfer(args):
    # Enable GPU
    use_cuda = enable_gpu(args)

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Resize(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
#     content = Variable(content).type(dtype)
    content = Variable(content)

    # load style model
#     style_model = ImageTransformNet().type(dtype)
    style_model = ImageTransformNet()
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
	parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

	style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    if (args.subcommand == "transfer"):
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")