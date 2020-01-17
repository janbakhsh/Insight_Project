Project overview & expectation
•	To automate the data extraction stage through
Segmentation of the documents into text classes such as:
•	body-text
•	titles/headings
•	footers/headers/pagination
•	tables
•	signatures/seals.

Approach
Using a combination of ML/DNN algorithms
i.	NLP-based classifiers 
i.	Struggle with segments like lists and bullet points
ii.	Model must be trained and tested on the same doc style

ii.	CNN-based classifiers 
i.	Intuitive like human vision classifier
ii.	Same model can be tested on any vocab document

Data
PDF files 
•	Single pdf pages  single jpg files 
•	Each page to be segmented into classes (labels)
Insert sample data images here …

Steps
i.	Automating the labeling stage given the short timeline for completion

ii.	Identifying the right model prior to training, due to computational expense of these models

iii.	Model training/validation, and testing on unseen document

iv.	Encapsulating the trained model into a user-friendly application

Model
In progress …

Validation
	20 legal documents to be segmented into multiple text sections
	Accuracy be measured by Recall, Precession, and F1score 
Application
Multi-object detection and classification in image and text files
