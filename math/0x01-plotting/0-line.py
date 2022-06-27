#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

y = np.arange(0, 11) ** 3

x = np.arange(11)
plt.plot(x, y, 'r-')
plt.axis([0, 10, -50, 1050])
plt.ylabel('some numbers')
plt.show()
