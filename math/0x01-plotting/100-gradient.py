#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)

# Data
x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# Plotting
cbar.set_label("elevation (m)")
plt.xlabel("x coordinate (m)")
plt.title("Mountain Elevation")
plt.ylabel("y coordinate (m)")
plt.scatter(x, y, c=z)
cbar = plt.colorbar()

plt.show()
