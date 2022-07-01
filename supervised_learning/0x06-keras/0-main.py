#!/usr/bin/env python3

# Force Seed - fix for Keras
SEED = 0

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import tensorflow.keras as K

build_model = __import__('0-sequential').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)