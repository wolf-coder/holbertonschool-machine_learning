#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# plotting
plt.plot(x, y1, 'r--', c='red', label='C-14')
plt.plot(x, y2, c='green', label='Ra-226')
plt.xlim(x[0], x[-1])
plt.ylim(min(y1[-1], y2[-1]), y1[0])
plt.legend()
plt.xlabel("Time (years)")
plt.ylabel("labeled Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")
plt.show()
