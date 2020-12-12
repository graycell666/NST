# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:58:16 2020

@author: sheng
"""
from __future__ import print_function

import torch

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import Loss
import Image_op
import transfer

def train_one_image(style_img, content_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # desired size of the output image
    #imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    
    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
    
    """
    plt.ion()
    
    plt.figure()
    Image_op.imshow(style_img, title='Style Image')
    
    plt.figure()
    Image_op.imshow(content_img, title='Content Image')
    """
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # desired depth layers to compute style/content losses :
    """
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1','conv_3','conv_5','conv_9','conv_13']
    
    """
    content_layers_default = ['conv_10']
    style_layers_default = ['conv_1','conv_3','conv_5','conv_9','conv_13']
    
    #input_img = style_img.clone()
    #input_img = torch.randn(content_img.data.size()).to(device, torch.float)
    input_img = content_img.clone()
    
    
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)
    
    # add the original input image to the figure:
    #plt.figure()
    #Image_op.imshow(input_img, title='Input Image')
    
    
    
    output, style_loss, content_loss = transfer.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, content_layers_default, style_layers_default)
    

    
    return output, style_loss, content_loss

    
    
    