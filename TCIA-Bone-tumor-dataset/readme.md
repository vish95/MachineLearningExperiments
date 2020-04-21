This project is an attempt to predict if a tumor is malignant or benign from the given features. 
The mat files attached contain extracted and simplified data to perform logistic regression/binary classification. 
The original dataset consists of 3 labels, which were simplified into two, labels indicating tumor whether viable or non-viable
were taken as positve and labels indicating no tumor or were taken as negative samples. The data was also shuffled and stored. 
This data is loaded into the matlab script and split into test, train and validation data. 

Since the feature list provided with a lot of data, it was first given preference as it contains extracted feature data from images.
Further image analysis and feature extraction was not carried out.

Logistic regression was carried out with the existing dataset.
With the obtained results it was observed that both train and validation accuracy was about 50%. This might be an indication that features
are not representative of the problem, new features may need to be extracted from the data or the images.

