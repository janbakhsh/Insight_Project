import os
import math
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
#import cv2
from fastai.vision import *

from fastai.metrics import error_rate 


def test1():
        #from utils.trainer_functions import *
    
#    img1 = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data/trainX_pngs/pdf_input_000_1-000001.png")
    
#    img1.show(figsize=(28, 16), title='Sample Train image')
    
    """Step 1: Determine where the training images are stored
    changing the resize_method in the data = block will change how the images are resized.  
    Your options are ResizeMethod.SQUISH and ResizeMethod.PAD"""
    
    root_path = Path(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data")
    seg_path = root_path/"trainX_pngs"
    codes_path = root_path/"codes_run1.txt"
    return root_path, seg_path, codes_path

root_path, seg_path, codes_path = test1()   

"""Step 2: Determine how to get the mask name from the image name"""
def get_y_lambda_function(root_path, mask_folder = 'trainY_pngs_norm_run1', image_folder = 'trainX_pngs'):
    
    return lambda filename: root_path/mask_folder/filename.relative_to(root_path/image_folder).parent/(filename.name.replace('input', 'target'))

def get_mask_func():
    root_path, seg_path, codes_path = test1()   
    get_mask_func = get_y_lambda_function(root_path)
    return get_mask_func

get_mask_func = get_mask_func()

    
"""Step 3: Change any of the default parameters:"""
lr = 2e-4                   #Learning Rate
bs = 4                      #Batch Size
size = 350 # 400                  #Size of the network (images reshaped to (sz, sz))
model = models.resnet34     #Model 
export_name = 'RandABHome.pkl'  #Export Name
wd = 1e-3
penalty = 1
tfms = get_transforms()
tfms = [[],[]]
    
# """Step 4: Detemine how you want you accuracy to be determined"""
# def acc_segmentation(input_image, target):
#     target = target.squeeze(1)
#     return (input_image.argmax(dim=1)==target).float().mean()


"""Step 5: Load the pixel value to feature codes"""

codes = np.loadtxt(codes_path, dtype = 'str')
name2id = {code:number for number,code in enumerate(codes)}


"""Step 6: Create your dataset"""
#Note: Use ResizeMethod.PAD to pad to a square and SQUISH to squish the larger axis
data = (SegmentationItemList
         .from_folder(seg_path, recurse = True)
         .split_by_rand_pct()
         .label_from_func (get_mask_func, classes = codes)
         .transform(tfms = tfms, size = size, tfm_y = True, padding_mode = 'border', resize_method = ResizeMethod.SQUISH)
         .databunch(bs = bs, num_workers = 0))

def test2():
    imported_learn = load_learner(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1", "Test_Model_run1.pkl", num_workers = 0)
    return imported_learn

def test3(imported_learn):
    
    img1 = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data/trainX_pngs/pdf_input_000_1-000001.png")
    
    img1 = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data/trainX_pngs/pdf_input_000_1-000001.png")
    
    img1.show(figsize=(28, 16), title='Test image')
    
    imported_learn.predict(img1)[0].show()
    
    imported_learn.predict(img1)[0].show(figsize=(28, 16), title='Predicted image')


def dummy():
    print('Hi')