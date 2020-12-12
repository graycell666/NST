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


# In[7]:


def visualize(args):
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Resize(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        testImage_amber = utils.load_image("content_imgs/amber.jpg")
        testImage_amber = img_transform_512(testImage_amber)
#         testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
        testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False)

        testImage_dan = utils.load_image("content_imgs/dan.jpg")
        testImage_dan = img_transform_512(testImage_dan)
#         testImage_dan = Variable(testImage_dan.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
        testImage_dan = Variable(testImage_dan.repeat(1, 1, 1, 1), requires_grad=False)

        testImage_maine = utils.load_image("content_imgs/maine.jpg")
        testImage_maine = img_transform_512(testImage_maine)
#         testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
        testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False)


# In[8]:


def transform(args, vgg):
    # Transform dataset
    dataset_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    # Transform style
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style)
#     style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1))
    style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # Calculate gram matrices for style features
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]
    return train_loader, style_gram, style_name


# In[9]:


def train(args, image_transformer, train_loader, optimizer, vgg, loss_mse, style_gram):
    for e in range(EPOCHS):
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read
            optimizer.zero_grad()
            
            # input batch to transformer network
#             x = Variable(x).type(dtype)
            x = Variable(x)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.data

            # calculate content loss
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.data

            # calculate total variation regularization
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.data

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # back propagation
            total_loss.backward()
            optimizer.step()

            # check the status for each 100 batches
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.data[0], content_loss.data[0], tv_loss.data[0]
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                outputTestImage_amber = image_transformer(testImage_amber).cpu()
                amber_path = "visualization/%s/amber_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(amber_path, outputTestImage_amber.data[0])

                outputTestImage_dan = image_transformer(testImage_dan).cpu()
                dan_path = "visualization/%s/dan_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(dan_path, outputTestImage_dan.data[0])

                outputTestImage_maine = image_transformer(testImage_maine).cpu()
                maine_path = "visualization/%s/maine_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(maine_path, outputTestImage_maine.data[0])

                print("images saved")
                image_transformer.train()


# In[10]:


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


# In[14]:


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    args = parser.parse_args()
    
    if (args.subcommand == "train"):
        print("Training!")
        # Enable GPU
        use_cuda = False
        if enable_gpu(args):
            use_cuda = True
        # Visualize the training
        visualize(args)
        # Define and load the network    
#         image_transformer = ImageTransformNet().type(dtype)
        image_transformer = ImageTransformNet()
        optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 
        loss_mse = torch.nn.MSELoss()
#         vgg = Vgg16().type(dtype)
        vgg = Vgg16()
        # Transform
        train_loader, style_gram, style_name = transform(args, vgg)
        train(args, image_transformer, train_loader, optimizer, vgg, loss_mse, style_gram)
        # save model
        image_transformer.eval()
        if use_cuda:
            image_transformer.cpu()
        if not os.path.exists("models"):
            os.makedirs("models")
        filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
        torch.save(image_transformer.state_dict(), filename)
        if use_cuda:
            image_transformer.cuda()
    else:
        print("invalid command")


# In[15]:


if __name__ == '__main__':
    main()


# In[ ]:




