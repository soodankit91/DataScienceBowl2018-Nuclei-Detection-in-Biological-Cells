from unet_model import  train_model, define_model, test_model, train_model_generator
from unet_postProcess import process_prediction
from unet_preprocess import read_data, generator

# Set some parameters
IMG_WIDTH = 256         #to set the dimensions of the image 
IMG_HEIGHT = 256        # to set the dimensions of the image 
                        #256 was chosen as an optimal value, higher values will result in increased computation cost,
                        #lower values will result in loss of information
IMG_CHANNELS = 4        # number of image channels
TRAIN_PATH = '../train/'    # path to training data, contains two folders, ‘images’ and ‘masks’ 
TEST_PATH = '../test/'      # path to test data, contains a folder ‘images’
batch_size = 32             # size of one batch of training data
validation_split_param=0.2  # Fraction of training data to be used for validation
epochs_param=50             # number of epochs i.e number of times the entire training set will be passed though the neural network. 

def main():
    X_train, Y_train, X_test, test_ids, sizes_test = read_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH)
    model = define_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # training without data augmentation
    model = train_model(X_train,Y_train, model, validation_split_param, batch_size, epochs_param)

    # comment the above statement and uncomment the below one for training with data augmentation

    #xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
    #train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size)
    #model = train_model_generator(model, train_generator, val_generator, xtr, xval, batch_size)

    preds_test_upsampled = test_model(model, X_test, sizes_test)
    process_prediction(preds_test_upsampled, test_ids)

main()