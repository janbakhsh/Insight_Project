Training, Validation, and Test sets are not provided in this folder. However, the details are outlined bellow: 

1- Data was in PDF format files, auto generated with coresponding anotated label files.
2- PDF files were converted to PNG images, creating 15000 pages for training and validation.
3- Of the above 15000 images those with descrepensy errors between input and target images were excluded from the training/val set.
4- Corrupted samples were exported to a separate folder for further investigation.
5- 3000 images were seleced for the training of the model (initial set).
6- Test set contained 4 pdf pages.
7- The first version of working model using the above 3000 training/val set is provided in notebook folder under the name:
Final_1_Model_3000samples_resnet34_10epochs.ipynb
