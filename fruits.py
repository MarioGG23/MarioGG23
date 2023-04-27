import os 																			                                # for manipulating the directories
import cv2 																			                                # for image processing 
import random 																		                              # for shuffling
import numpy as np 																	                            # for array manipulating and scientific computing
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

import tensorflow as tf 															                          # for more details see: https://www.tensorflow.org/tutorials
from tensorflow import keras 														                        # for more details see: https://www.tensorflow.org/guide/keras/overview

from tensorflow.keras.models import Model 								                      # for more details see about Model class API: https://keras.io/models/model/
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical				       				          # for categorical labels
from tensorflow.keras import optimizers
import random
# general parameters
NAME = 'fruits-classifier'                                                      # name for the callback output
CATEGORIES = ["Apple Golden 1","Apple Pink Lady","Apple Red 1","Pear Red","Pear Williams","Pear Monster"] 	# we work with three classes of Apple and Pear
class_names = CATEGORIES
num_classes = 6
img_size = 100

base_dir = 'Fruit-Images-Dataset/'

# Read training set
train_images = []
train_dir = os.path.join(base_dir, 'Training/')										              # set the training directory in the path

for category in CATEGORIES:															                        # iterate to each category
    path = os.path.join(train_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):													                    # iterate to each image in the category
        if(image.endswith('jpg') and not image.startswith('.')):
            img_array = cv2.imread(os.path.join(path,image),                    # read the image
                              cv2.IMREAD_GRAYSCALE)	
            train_images.append([img_array, class_num])								          # save the image in training data array

print("Training images: ", len(train_images))



# Read testing set
test_images = []
test_dir = os.path.join(base_dir, 'Test/')											                # set the test directory in the path

for category in CATEGORIES:															                        # iterate to each category
    path = os.path.join(test_dir, category)
    class_num = CATEGORIES.index(category)
    for image in os.listdir(path):													                    # iterate to each image in the category
        if(image.endswith('jpg') and not image.startswith('.')):													
            img_array = cv2.imread(os.path.join(path,image),                    # read the image
                                   cv2.IMREAD_GRAYSCALE)	
            test_images.append([img_array, class_num])								          # save the image in test data array
            
print("Testing images: ", len(test_images))

# Shuffle the dataset before training for better accuracy
x_train = []																		                                # array for images
y_train = []																		                                # array for labels

random.shuffle(train_images)														                        # shuffle training images

for features, label in train_images: 												                    # iterate to each image and the corresponding label in training data
	x_train.append(features)
	y_train.append(label)
x_train = np.array(x_train)
 
x_test = []																			                                # array for images
y_test = []																			                                # array for labels

random.shuffle(test_images) 														                        # shuffle testing images

for features, label in test_images: 												                    # iterate to each image and the corresponding label in training data
	x_test.append(features)
	y_test.append(label)
x_test = np.array(x_test)

# reshape and normalize the data before training
x_train = x_train.reshape(-1, img_size, img_size, 1)
mean_train = np.mean(x_train, axis=0)
x_train = x_train-mean_train
x_train = x_train/255

x_test = x_test.reshape(-1, img_size, img_size, 1)
mean_test = np.mean(x_test, axis=0)
x_test = x_test-mean_test
x_test = x_test/255

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print(x_train.shape)
print(x_test.shape)

# Hyperparameters settings
input_shape = x_train.shape[1:]
filters_numbers = [16, 32, 64]
filters_size = [[5,5],[4,4],[3,3]]

pool_size=(2, 2)
weight_decay = 5e-4
dropout = 0.6
lr = 0.001
momentum = 0.9

epochs = 10
batch_size = 32

L2_norm = keras.regularizers.l2(weight_decay)

# Setup model layers

# Input layer
model_input = Input(shape=input_shape)

# 1st Convolutional layer
model_output = Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), kernel_regularizer=L2_norm, padding="Same", 
							activation='relu', data_format='channels_last')(model_input)

model_output = BatchNormalization()(model_output)

model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

# 2nd Convolutional layer
model_output = Conv2D(filters_numbers[1], kernel_size=(filters_size[1]), kernel_regularizer=L2_norm, padding="Same",  
							activation='relu', data_format='channels_last')(model_output)

model_output = BatchNormalization()(model_output)

model_output = MaxPooling2D(pool_size=(pool_size))(model_output)

# 3rd Convolutional layer
model_output = Conv2D(filters_numbers[2], kernel_size=(filters_size[2]), kernel_regularizer=L2_norm, padding="Same",  
							activation='relu', data_format='channels_last')(model_output)

model_output = BatchNormalization()(model_output)

model_ouput = GlobalAveragePooling2D(data_format='channels_last')(model_output)

# Convert features to flatten vector      
model_output = Flatten()(model_output)

# Full-connected layer
model_output = Dense(512)(model_output)
model_output = Dropout(dropout)(model_output)

# Output layer
model_output = Dense(num_classes, activation='softmax', name='id')(model_output)

# Create the Model by using Input and Output layers
model = Model(inputs=model_input, outputs=model_output, name=NAME)

# Show the Model summary information
model.summary()

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizers.SGD(lr, momentum), metrics=['accuracy'])

# Train the model
print("[INFO] Train the model on training data")

history = model.fit(x=x_train, y=np.asarray(y_train), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# Test the model
print("[INFO] Evaluate the test data")

results = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
print('Testing Loss, Testing Acc: ', [round(r,4) for r in results])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print('Accuracy', round(accuracy_score(y_test, y_pred),4))
print('Classification report', classification_report(y_test, y_pred, target_names=class_names))