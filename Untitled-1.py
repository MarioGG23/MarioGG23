import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import random

theSEED = 232323
tf.random.set_seed(theSEED)
np.random.seed(theSEED)
random.seed(theSEED)

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train[0].shape

import matplotlib.pyplot as plt
#plt.imshow(x_train[0])
print(y_train[0])

x_train  = x_train / 255.0
x_test = x_test / 255.0

# Hiperparámetros: parámetros no optimizables por el algoritmo de retropropagación
# Suelen fijarse mediante ensayo y error y mediante experiencia

NB_EPOCH = 10#100       # numero de epocas de entrenamiento
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_split=VALIDATION_SPLIT)

model.evaluate(x_test, y_test)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
