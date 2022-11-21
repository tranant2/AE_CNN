# AE_CNN

This code was used for one of my research paper and this is the public version for reference. 
This code consist of CNN9 jupyter notebook
CNNModels.py, CNNTrain.py, and H5Dataset.py which are python files which contains the various functions used for the paper
CNN9.py, CNN9_.py, and CNN9__.py which are three different files used for training the model.
The rest are various outputs from the code and model parameters.

## CNN9 Jupyter Notebook
Start here. This goes over the main code used to train the whole model

## CNNModels.py
This contains the different networks such as the encoder/decoder network, and the big NN which combines the encoder/decoder and loss evaluator

## CNNTrain.py
This describe how the training procedure was done. One thing to note is that the loss function was changed for the three different training procedures in order to  only train selected layers.

## H5Dataset.py
This code is a custom dataset used to input the data in the model. The data used is in the model is an hdf5 format. 

## CNN9.py, CNN9_.py, and CNN9__.py
These are the different code used to train the model. In CNN9, part of the model was frozen while in CNN9_, the other part of the model was frozen. Finally, in CNN9__, the whole model was unfrozen and trained with a smaller learning rate.

