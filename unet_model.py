import numpy as np
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf


def define_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

	inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	s = Lambda(lambda x: x / 255) (inputs)

	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', ) (s)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.1) (c8)
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[inputs], outputs=[outputs])

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

	return model

def train_model(X_train,Y_train, model, validation_split_param=0.2, batch_size_param=32, epochs_param=50):
	model.fit(X_train, Y_train, validation_split=validation_split_param, batch_size=batch_size_param, epochs=epochs_param)

	return model

def train_model_generator(model,train_generator, val_generator, xtr, xval, batch_size , epochs_param = 50 ):
	model.fit_generator(train_generator, steps_per_epoch=len(xtr) / 6, epochs=epochs_param, validation_data=val_generator, validation_steps=len(xval)/batch_size)

	return model

def test_model(model, X_test, sizes_test):
	preds_test = model.predict(X_test, verbose=1)

	# Create list of upsampled test masks, since all test images were scaled to 256*256,
	# they need to be brought to their original dimensions before run length encoding can be done
	preds_test_upsampled = []
	for i in range(len(preds_test)):
	    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))

	return preds_test_upsampled


# IoU metric
def mean_iou(y_true, y_pred):
	prec = []
	for t in np.arange(0.5, 1.0, 0.05):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# another loss metric which can be used in place of binary_crossentropy
def bce_dice_loss(y_true, y_pred):
	return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)