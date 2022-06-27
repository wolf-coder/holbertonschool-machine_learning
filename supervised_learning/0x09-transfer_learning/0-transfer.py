#!/usr/bin/env python3
"""
a python script that trains a convolutional neural network to classify the CIFAR 10 dataset
"""
import keras as K
import numpy as np
from sklearn.utils.multiclass import unique_labels
import os
from sklearn.model_selection import train_test_split

from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import gradient_descent_v2, adam_v2, rmsprop_v2
from keras.backend import resize_images
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout, Lambda

from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize_with_pad

"""Import dataset"""
from keras.datasets import cifar10


def preprocess_data(X, Y):
  """pre-processes the data for the 
  X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data
  Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
  Returns: X_p, Y_p
  X_p: numpy.ndarray containing the preprocessed X
  Y_p: numpy.ndarray containing the preprocessed Y"""
  y_one_hot = to_categorical(Y)
  X_p = K.applications.densenet.preprocess_input(X.astype('float32'))
  Y_p = y_one_hot
  return X_p, Y_p

if __name__ == "__main__":
  """
  python script that trains a convolutional neural network to classify the CIFAR 10 dataset
  """
  """Load data and preprocess it"""
  (x_train, y_train), (x_test, y_test)= cifar10.load_data()
  x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,test_size=.3)
  x_train, y_train = preprocess_data(x_train, y_train)
  x_val, y_val =preprocess_data(x_val, y_val)
  x_test, y_test = preprocess_data(x_test, y_test)

  """ Data augmentation (ImageDataGenerator)"""
  train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
  val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
  test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True, zoom_range=.1)

  train_generator.fit(x_train)
  val_generator.fit(x_val)
  test_generator.fit(x_test)

  """ building the model"""
  base_model = DenseNet121(include_top=False,weights='imagenet',classes=y_train.shape[1],pooling='max')
  base_model.trainable = False

  model= K.Sequential()
  model.add(Lambda(lambda x:K.preprocessing.image.smart_resize(x, (160, 160))))
  model.add(base_model)
  model.add(Flatten())
  """Adding the Dense layers along with activation and batch normalization"""
  model.add(BatchNormalization())
  model.add(Dense(256,activation=('relu'))) 
  model.add(Dropout(.3))
  model.add(BatchNormalization())
  model.add(Dense(128,activation=('relu')))
  model.add(Dropout(.2))
  model.add(BatchNormalization())
  model.add(Dense(64,activation=('relu')))
  model.add(Dropout(.2))
  """Adding the classification layer"""
  model.add(Dense(10,activation=('softmax')))

  batch_size= 128
  epochs=20
  learning_rate=.001
  sgd= gradient_descent_v2.SGD(learning_rate=learning_rate,momentum=.9,nesterov=False)
  adam = adam_v2.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  rms = rmsprop_v2.RMSProp(learning_rate=learning_rate)

  callbacs = []
  callbacs.append(K.callbacks.ModelCheckpoint(filepath='cifar10.h5', monitor='val_accuracy', save_best_only=True))
  callbacs.append(K.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=5))
  callbacs.append(K.callbacks.TensorBoard(log_dir='logs'))

  """Compiling the model"""
  model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
  """training the model"""
  model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data=(x_val, y_val), callbacks=callbacs)
  """saving and evaluating the model"""
  model.save("cifar10.h5")
  model.evaluate(x_test, y_test, batch_size=128, verbose=1)