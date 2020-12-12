# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:30:32 2020

@author: sheng
"""
import torch
import torch.nn as nn
from torchvision import transforms

import train
import Image_op
import PIL

import matplotlib.pyplot as plt

import time
import pandas as pd

import os


if __name__ == '__main__':
    
    
    
    img_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    df = pd.DataFrame(columns=['style_img', 'content_class','content_img','time','styleloss', 'contentloss'])
    
    
    style_folder = './test/testing/style'
    content_folder = './test/testing/content'
    
    output_folder = './test/testing/output'
    
    if (not os.path.exists(output_folder)):
        os.makedirs(output_folder)
    
    for style_img_name in os.listdir(style_folder):
        if(not style_img_name == 'EUCiVwEXYAEFH_2.jpg'):
            continue

        style_img_path = os.path.join(style_folder, style_img_name)
        style_img = Image_op.image_loader(style_img_path, img_size, device)
        
        style_output_folder = os.path.join(output_folder, style_img_name)
        
        if (not os.path.exists(style_output_folder)):
            os.makedirs(style_output_folder)

        
        for content_category in os.listdir(content_folder):
            
            content_category_path = os.path.join(content_folder, content_category)
            output_path = os.path.join(style_output_folder, content_category)
            
            if (not os.path.exists(output_path)):
                os.makedirs(output_path)
                
            for content_img_name in os.listdir(content_category_path):

                content_img_path = os.path.join(content_category_path, content_img_name)
                content_img = Image_op.image_loader(content_img_path, img_size, device)


                
                start = time.time()
                output_image, style_loss, content_loss = train.train_one_image(style_img, content_img) #2 images must be the same size here
                end = time.time()
                print(end - start)
                
                new_row = {'style_img':style_img_name, 'content_class':content_category, 'content_img':content_img_name, 
                           'time':end - start, 'styleloss':style_loss, 'contentloss':content_loss}
                
                df = df.append(new_row, ignore_index=True)
                #df.info()
                
                output_image_PIL = Image_op.convert_to_PIL(output_image)
                
                output_image_PIL.save(os.path.join(output_path, content_img_name), 'png')
                
    df.to_excel(os.path.join(output_folder, 'log.xlsx'))
    df.to_csv(os.path.join(output_folder, 'log.csv'), index=False)
                
                
                
    """
    style_img = Image_op.image_loader('EUCiVwEXYAEFH_2.jpg', img_size, device)
    content_img = Image_op.image_loader("03MQCT6SCX.jpg", img_size, device)
    
    #plt.figure()
    #Image_op.imshow(style_img, title='Style Image')
    
    start = time.time()
    output_image, style_loss, content_loss = train.train_one_image(style_img, content_img) #2 images must be the same size here
    end = time.time()
    print(end - start)
    
    #new_row = {'style_img':'', 'content_img':'', 'time':end - start}
    #df = df.append(new_row, ignore_index=True)

    #save PIL image
    output_image_PIL = Image_op.convert_to_PIL(output_image)
                
    output_image_PIL.save('03MQCT6SCX_conv_10.png', 'png')
    
    # sphinx_gallery_thumbnail_number = 4
    #plt.ioff()
    #plt.show()"""