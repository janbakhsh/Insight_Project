
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

from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image

from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)



root_path = Path(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data")
seg_path = root_path/"trainX_pngs"
codes_path = root_path/"codes_run1.txt"

"""Step 2: Determine how to get the mask name from the image name"""
def get_y_lambda_function(root_path, mask_folder = 'trainY_pngs_norm_run1', image_folder = 'trainX_pngs'):
    return lambda filename: root_path/mask_folder/filename.relative_to(root_path/image_folder).parent/(filename.name.replace('input', 'target'))

get_mask_func = get_y_lambda_function(root_path)

"""Step 4: Detemine how you want you accuracy to be determined"""
def acc_segmentation(input_image, target):
    target = target.squeeze(1)
    return (input_image.argmax(dim=1)==target).float().mean()

""" LOAD THE TRAINED MODEL """

imported_learn = load_learner(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1", "Test_Model_run1.pkl", num_workers = 0)

#img1 = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data/trainX_pngs/pdf_input_000_1-000001.png")

img1 = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data/trainX_pngs/pdf_input_000_13-000039.png")
img1.show(figsize=(28, 16), title='Test image')


def predict_single(img_file):
    'function to take image and return prediction'
    segmented_image = imported_learn.predict(img1)[0]
    segmented_image.show(figsize=(28, 16), title='Predicted image')
  
predict_single(img1)






