# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:44:10 2020

@author: sheng
"""
import time
import pandas as pd

import os

#test code for fast NST

if __name__ == '__main__':
    df = pd.DataFrame(columns=['style_img', 'content_class','content_img','time'])
       
    style_model_folder = './models/models/'
    content_folder = './test_content/'
    output_folder = './output/'
    
    python_script = './fast_neural_style/neural_style/neural_style.py'
    
    if (not os.path.exists(output_folder)):
        os.makedirs(output_folder)
    
    for style_model_name in os.listdir(style_model_folder):
        style_model_subfolder = os.path.join(style_model_folder, style_model_name)
        if (style_model_name == 'lisa.pth'):
            continue
        
        print('eval: ', style_model_name)
        path = os.listdir(style_model_subfolder)[0]
        style_model_path = os.path.join(style_model_subfolder+'/', path)
        
        style_output_folder = os.path.join(output_folder, style_model_name)
        
        if (not os.path.exists(style_output_folder)):
            os.makedirs(style_output_folder)
    
        for content_category in os.listdir(content_folder):
            content_category_path = os.path.join(content_folder, content_category)
            output_path = os.path.join(style_output_folder+'/', content_category)
            
            if (not os.path.exists(output_path)):
                os.makedirs(output_path)
                
            print('eval: ', content_category)
            for content_img_name in os.listdir(content_category_path):

                content_img_path = os.path.join(content_category_path+'/', content_img_name)
                
                output_img_path = os.path.join(output_path+'/', content_img_name)
                
                start = time.time()
                
                #output_image, style_loss, content_loss = train.train_one_image(style_img, content_img) #2 images must be the same size here
                #python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
                
                command = 'python '+python_script+' eval --content-image '+content_img_path+' --model '+style_model_path+' --output-image '+output_img_path+' --cuda 1'
                os.system(command)
                #subprocess.call(['runas', '/user:Administrator', command])
                #print(command)
                end = time.time()
                print(end - start)
                
                new_row = {'style_img':style_model_name, 'content_class':content_category, 'content_img':content_img_name, 
                           'time':end - start}
                
                df = df.append(new_row, ignore_index=True)
                #df.info()
                
    df.to_excel(os.path.join(output_folder, 'log.xlsx'))
    df.to_csv(os.path.join(output_folder, 'log.csv'), index=False)