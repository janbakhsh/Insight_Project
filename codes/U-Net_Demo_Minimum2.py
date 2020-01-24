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
# See your current version of python/anaconda
print (sys.version)
#from utils.trainer_functions import *

###########################################################################################################


# Normalize the masks...
def normalize_images(root_path, mask_folder, save_folder):
    #root_path = Path(r"C:\Users\William\Desktop\data - Copy\cleaned")
    root_path = Path(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data")
    mask_path = root_path/'trainY_pngs'#Path(r"C:\Users\Rocheleau Microscope\.fastai\data\Mask Trainer\Mask\INS1E\20190307W")
    save_path = root_path/'trainY_pngs_norm_dummy'

    save_path.mkdir(parents = True ,exist_ok = True)
    
    colours = [([255,255,255], 0),
               ([255,0,0], 1),
               ([0,0,255], 2),
               ([255,242,0], 3),
               ([0,255,0],4),
          ]


    for files in mask_path.iterdir():
        reader = imageio.get_reader(str(files))
        img = reader.get_data(0)
        norm_img = np.zeros(img.shape[:2], dtype = np.uint8)
        print(norm_img.shape)
        
        
        for (colour, value) in colours:
            r,g,b = colour
            colours_match = np.all([img[:,:,0] == r, img[:,:,1] == g, img[:,:,2] == b], axis=0)
            norm_img[colours_match] = value




        imageio.imwrite(str(save_path/files.name), norm_img)
        #print(np.amax(img))


###########################################################################################################
"""Only run the below code if you need to normalize your masks"""
normalize_images(root_path = r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data", 
                 mask_folder = 'trainY_pngs', 
                 save_folder = 'trainY_pngs_norm_dumy')        
        
###########################################################################################################
"""Step 1: Determine where the training images are stored"""
root_path = Path(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data")
seg_path = root_path/"trainX_pngs"
codes_path = root_path/"codes_run1.txt"     
        
        
###########################################################################################################        
"""Step 2: Determine how to get the mask name from the image name"""
def get_y_lambda_function(root_path, mask_folder = 'trainY_pngs_norm_dumy', image_folder = 'trainX_pngs'):
    return lambda filename: root_path/mask_folder/filename.relative_to(root_path/image_folder).parent/(filename.name.replace('input', 'target'))

get_mask_func = get_y_lambda_function(root_path)        
###########################################################################################################
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
        
        
###########################################################################################################
"""Step 4: Detemine how you want you accuracy to be determined"""
def acc_segmentation(input_image, target):
    target = target.squeeze(1)
    return (input_image.argmax(dim=1)==target).float().mean()      
        
        
###########################################################################################################
"""Step 5: Load the pixel value to feature codes"""
codes = np.loadtxt(codes_path, dtype = 'str')
name2id = {code:number for number,code in enumerate(codes)}     
        
###########################################################################################################
# changing the resize_method in the data = block will change how the images are resized.options are ResizeMethod.SQUISH and ResizeMethod.PAD              
"""Step 6: Create your dataset"""
#Note: Use ResizeMethod.PAD to pad to a square and SQUISH to squish the larger axis
data = (SegmentationItemList
         .from_folder(seg_path, recurse = True)
         .split_by_rand_pct()
         .label_from_func (get_mask_func, classes = codes)
         .transform(tfms = tfms, size = size, tfm_y = True, padding_mode = 'border', resize_method = ResizeMethod.SQUISH)
         .databunch(bs = bs, num_workers = 0))     
        
###########################################################################################################
"""Step 7: Train your learner """
learn = unet_learner(data, model, wd = wd, metrics = [acc_segmentation])
                 #loss_func = CrossEntropyFlat(axis=1, weight = torch.FloatTensor([1,penalty]).cuda())
                  #  )       
###########################################################################################################
""" Train the model with most of the layers frozen """
###########################################################################################################

"""Find the appropriate learning rate by choosing an area of continuous descent"""

learn.lr_find()
learn.recorder.plot()
#The results indicate that 
###########################################################################################################
data.show_batch()

###########################################################################################################
learn.fit_one_cycle(5, max_lr = lr)

###########################################################################################################
learn.show_results()

###########################################################################################################
""" Next we unfreeze the rest of the layers and retrain using a new learning rate """
###########################################################################################################

"""When we unfreeze the layers, they can all be trained"""
learn.unfreeze()

###########################################################################################################
"""We must choose a new learning rate to optimize training"""

learn.lr_find(start_lr = 1e-4)
learn.recorder.plot()
#results suggest 

###########################################################################################################
learn.fit_one_cycle(20, max_lr = lr/10)

###########################################################################################################
learn.show_results()

###########################################################################################################

learn.export(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Test_Model_dummy.pkl")

###########################################################################################################


###########################################################################################################

# imported_learn = load_learner(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1", "Test_Model_dumy.pkl", num_workers = 0)

imported_learn = load_learner(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1", "Test_Model_run1.pkl", num_workers = 0)
###########################################################################################################
img = open_image(r"C:\Users\payman\Documents\Insight_Project\scrapworks\Test_Models\Model1\Data\trainX_pngs\pdf_input_000_5-000001.png")

###########################################################################################################
img.show()
predicted_img = 
imported_learn.predict(img)[0]

###########################################################################################################
imported_learn.predict(img)[0].show()
# might have to downgrade fastai to 1.0.55
###########################################################################################################

"""
Unet Network Training Experimentation

The goal of this program is to easily be able to compare the results of different combinations of labels in order to get a mask segmentation of the cells for later use.  It begins with the output of either the YOLO or the SSD network (simulated by manual ROI selection) and then uses the outputs from the SubimageGeneration section combined with manually created ground truths.  A future iteration will involve the ability to combine individual segmentations back into the main image, and then introduce a "Main Cell" vs "Auxillary Cell" class difference and the change the loss function to accomodate it (giving a much higher loss to the classifications of pixels as "Main Cell" when they are actually "Auxillary Cells").  If it works really well but doesn't distinguish well between the main cells and auxillary cells, I can always delete the segmentation for the auxillary cells manually and build my training dataset that way. 



In the open function, there will be a default behaviour for channels under 20 and the custom behaviour for other channels.  That will allow me to introduce interesting features like merging and randomness


The combinations that I'm planning on testing are as follows:
- Three Brightfield Channels (-10, 0, 10), (-7, 0, 7), (-3, 0, 3)
- Two Brightfield Channels and One Fluorescence
- One Brightfield and Two Fluorescence
- Two Brightfield and a merge of all of the fluorescence channels (add a special method to the open files function if the channel is above 16 (ex. 100 means merge all channels) - this will reduce the impact of a specific channel not being labelled


There will also be a randomness componenent added, which I will introduce by overwriting the channels above 100.  The channels will include:
- Two brightfield and one random fluorescence
- One "Below Focus" Image, One in focus, and One "Above Focus"
- Three Randomly Chosen Brightfield Images (sort them afterwards.  Duplicates allowed??? - Harder to implement with the current approach)



Other factors I can train are size

### Links to the relevant cells:
- [Core Helper Functions](#core_helper): Always run these cells before beginning!
- [General Training Functions](#general_training): This section will be your general trainer for a single model
- [Training Tester](#training_tester): This section will allow you to test many different combinations at once
- [Model Tester](#model_tester): Explore Aspects of your trained model (Work in progress...)
- [Transformation Tester](#transformation_tester): This section will allow you to look at the results of various transformations


"""














