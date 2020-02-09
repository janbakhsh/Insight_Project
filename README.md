# Project overview 

Object segmentation and text extraction from PDF files, such as:  
body-text  
Lists  
titles/headings  
footers/headers/pagination  
tables  
signatures/seals   
...  


# Approach

Applying computer vision models based on Pytorch framework  
i.	Usied pixel segmentation approach with U_net architecture, where Resnet34 was the encoder block  
ii.	Trained the model with multiple transforms such as random flips of 90 degrees, random rotations -10-10 degrees, applying perspective, zoom, lighting, and watermarks. The training took place at multiple stages including mutiple learning rate assesments, frozen and unfrozen weights of resnt34, and different image sizes such as 400x400 upto 800x800. These steps were applied on 230, 750, 3000, and 5000-size datasets  
iii. Accuracy was measured and reported for all classes present in the image (i.e, pdf document), and the overall accuracy while excluding the background  
iv. Two stages of transfer learning were applied , one using resnet's pretrained weights, and another using a well performed model on 400x400 images to data bunches with 800x800-size input images  
v.	Best performing model was tested on multiple unseen pdf pages (converted to PNG) containing title, subtitle, list, body, header/footer, and extracted contents (classes) were transfered to directories corresponding to their input image  
vi. Multi-language OCR (Optical Character Recognition) was applied to PNG images to extract text and transfer them to TXT format  

# Data

PDF files  
•	Different metrics were used to clean the dataset from any possible bad samples, where there was any degree of descrepency between input and label pixel segmentations   
•	Input and label PDF files were converted to PNG files, labels were normalized prior to training  

# Model training and validation accuracy

Class accuracies varried from 99% (for Body class) to 80% (for Title class)   

# Test stage

The script generates directories with names corresponding to each input test image  
Test PNG images (converted from PDF pages) were passed to the saved model and the predicted images were overlaid on   
the corresponding input images, and classified (detected) segments (title, subtitle, list, body, header/footer) for each image were saved in the corresponding directory for that image  
