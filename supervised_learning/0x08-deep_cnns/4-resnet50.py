#!/usr/bin/env python3
""" Resnet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition """
    init = K.initializers.he_normal()
    # input shape (224, 224, 3)
    X = K.Input((224, 224, 3))

    conv1 = K.layers.Conv2D(64, (7, 7),
                            padding='same', strides=2,
                            kernel_initializer=init)(X)
    b_norm1 = K.layers.BatchNormalization()(conv1)
    acti1 = K.layers.Activation('relu')(b_norm1)
    maxpool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                     padding='same')(acti1)

    proj = projection_block(maxpool1, [64, 64, 256], 1)
    iden = identity_block(proj, [64, 64, 256])
    iden = identity_block(iden, [64, 64, 256])
    proj = projection_block(iden, [128, 128, 512], 2)
    iden = identity_block(proj, [128, 128, 512])
    iden = identity_block(iden, [128, 128, 512])
    iden = identity_block(iden, [128, 128, 512])
    proj = projection_block(iden, [256, 256, 1024], 2)
    iden = identity_block(proj, [256, 256, 1024])
    iden = identity_block(iden, [256, 256, 1024])
    iden = identity_block(iden, [256, 256, 1024])
    iden = identity_block(iden, [256, 256, 1024])
    iden = identity_block(iden, [256, 256, 1024])
    proj = projection_block(iden, [512, 512, 2048], 2)
    iden = identity_block(proj, [512, 512, 2048])
    iden = identity_block(iden, [512, 512, 2048])

    avgpool = K.layers.AveragePooling2D((7, 7))(iden)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(avgpool)

    return K.models.Model(inputs=X, outputs=Y)
