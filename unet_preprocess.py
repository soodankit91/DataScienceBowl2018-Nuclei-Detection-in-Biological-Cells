import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator

# Rescaling images
def scale_img_canals(an_img,IMG_CHANNEL_COUNT=4):
    if IMG_CHANNEL_COUNT == 1:
        canal = an_img[:,:]
        canal = canal - canal.min()
        canalmax = canal.max()
        if canalmax > 0:
            factor = 255/canalmax
            canal = (canal * factor).astype(int)
            an_img[:,:] = canal
        return an_img

    for i in range(IMG_CHANNEL_COUNT):
        canal = an_img[:,:,i]
        canal = canal - canal.min()
        canalmax = canal.max()
        if canalmax > 0:
            factor = 255/canalmax
            canal = (canal * factor).astype(int)
            an_img[:,:,i] = canal
    return an_img

def read_data(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH):
    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,4), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        if(img.all()!=None):
            img = scale_img_canals(img)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        IMG_CHANNELS = 1
        try:
            IMG_CHANNELS = (imread(path + '/images/' + id_ + '.png').shape)[2]
            img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        except:
            img = imread(path + '/images/' + id_ + '.png')[:,:]

        sizes_test.append([img.shape[0], img.shape[1]])
        #img = scale_img_canals(img)
        if(img.all()!=None):
            img = scale_img_canals(img, IMG_CHANNELS)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH,4), mode='constant', preserve_range=True)
        X_test[n] = img

    print('Done!')
    return X_train, Y_train, X_test, test_ids, sizes_test

#Define generator. Using keras ImageDataGenerator. You can 
#change the method of data augmentation by changing data_gen_args.
def generator(xtr, xval, ytr, yval, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr, seed=7)
    mask_datagen.fit(ytr, seed=7)
    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval, seed=7)
    mask_datagen_val.fit(yval, seed=7)
    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator