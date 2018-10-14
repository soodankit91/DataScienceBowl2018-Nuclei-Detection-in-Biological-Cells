# DataScienceBowl2018-Nuclei-Detection-in-Biological-Cells

This is a solution to Kaggle's Data Science Bowl 2018, where the task was to identify Nuclei in biological cells 

For a detailed description of the solution, please refer 'solution.txt'

Link to the competition : https://www.kaggle.com/c/data-science-bowl-2018
Link to the dataset : https://www.kaggle.com/c/data-science-bowl-2018/data

unet_driver.py contains the main method. Executing the file will run the code.
Set the following parameters:
  IMG_WIDTH, IMG_HEIGHT : to set the dimensions of the image
  IMG_CHANNELS : number of image channels
  TRAIN_PATH, TEST_PATH : paths of training and test data respectively TRAIN_PATH contains two folders, ‘images’ and ‘masks’ TEST_PATH contains a folder ‘images’
  batch_size : size of one batch of training data
  validation_split_param : Fraction of training data to be used for validation
  epochs_param : number of epochs i.e number of times the entire training set will be passed though the neural network. 
