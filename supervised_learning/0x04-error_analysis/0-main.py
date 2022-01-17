#!/usr/bin/env python3

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)
"""Expected:
alexa@ubuntu-xenial:0x04-error_analysis$ ./0-main.py 
[[4701.    0.   36.   17.   12.   81.   38.   11.   35.    1.]
 [   0. 5494.   36.   21.    3.   38.    7.   13.   59.    7.]
 [  64.   93. 4188.  103.  108.   17.  162.   80.  132.   21.]
 [  30.   48.  171. 4310.    2.  252.   22.   86.  128.   52.]
 [  17.   27.   35.    0. 4338.   11.   84.    9.   27.  311.]
 [  89.   57.   45.  235.   70. 3631.  123.   33.  163.   60.]
 [  47.   32.   87.    1.   64.   83. 4607.    0.   29.    1.]
 [  26.   95.   75.    7.   58.   18.    1. 4682.   13.  200.]
 [  31.  153.   82.  174.   27.  179.   64.    7. 4003.  122.]
 [  48.   37.   39.   71.  220.   49.    8.  244.   46. 4226.]]
"""
