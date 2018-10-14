# DataScienceBowl2018-Nuclei-Detection-in-Biological-Cells

This is a solution to Kaggle's Data Science Bowl 2018, where the task was to identify Nuclei in biological cells images

Problem Description: Given several segmented nuclei images of biological cells and multiple images of masks per cell image, depicting the cell’s nuclei, the task is to train a model which would detect nuclei in unseen images of cells and report them as run length encoded pixels. 

Approach: Using a state of the art neural network for biological images, UNET for identification of nuclei in the test data.


For a detailed description of the solution, please refer 'Approach'

Link to the competition : https://www.kaggle.com/c/data-science-bowl-2018
Link to the dataset : https://www.kaggle.com/c/data-science-bowl-2018/data

Executing the code:
unet_driver.py contains the main method. Executing the file will run the code. Parameters can be set in the same file.
