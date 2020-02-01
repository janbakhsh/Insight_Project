Project overview 

•	Object segmentation and extraction from PDF files, such as:
•	body-text
•	titles/headings
•	footers/headers/pagination
•	tables
•	signatures/seals 
...


Approach
Applying computer vision models based on Pytorch framework
i.	Pixel segmentation using Unet architecture with Resnet34 in the encoder block
ii.	Trained the model with initial 3000 samples with accuracy measured based on overall correct pixel classification 
iii.	Saved model was tested on 4 unseen pdf pages containing title, subtitle, list, body, header/footer. same marging and layout format as in the training set  

Data
PDF files 
•	Input and target PDF files converted to PNG files, traget images (masks) were normalized prior to trainig
•	Total training/validation images were 3000

Model training and validation accuracy
Model was tarined with 41 million trainable prameters on Google Colab for 10 epochs. Overfitting appeared after 8 epochs and the pixel segmentation accuracy on validation set reached 99.32%

Test stage
The script generates directories with names corresponding to each input test image 
Test PNG images (converted from PDF pages) were passed to the saved model and the predicted image was overlaid on the input image and classified (detected) segments (title, subtitle, list, body, header/footer) for each image were saved in the corresponding directory for that image
