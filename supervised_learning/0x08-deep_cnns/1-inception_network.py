#!/usr/bin/env python3
"""inception network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network as"""
    hn = K.initializers.HeNormal()
    input_ = K.Input(shape=(224, 224, 3))
    cv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                          padding="same",
                          activation="relu",
                          kernel_initializer=hn)(input_)
    maxpool = K.layers.MaxPool2D((3, 3), 2, "same")(cv1)
    cv2 = K.layers.Conv2D(192, (3, 3),
                          padding="same",
                          activation="relu",
                          kernel_initializer=hn)(maxpool)
    maxpool1 = K.layers.MaxPooling2D((3, 3), 2, 'same')(cv2)
    inception3a = inception_block(maxpool1, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    maxpool2 = K.layers.MaxPooling2D((3, 3), (2, 2), "same")(inception3b)
    inception4a = inception_block(maxpool2, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    maxpool3 = K.layers.MaxPooling2D((3, 3), 2, 'same')(inception4e)
    inception5a = inception_block(maxpool3, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AveragePooling2D((7, 7))(inception5b)
    dropout = K.layers.Dropout(0.4)(avgpool)
    Y = K.layers.Dense(1000, activation='softmax')(dropout)
    return K.models.Model(inputs=input_, outputs=Y)
