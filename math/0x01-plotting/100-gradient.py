import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))


# Create a scatter plot using a colormap
scatter_plot = plt.scatter(x, y, s=40, c=z, cmap='viridis', marker='o', alpha=0.7)

# Add a colorbar to show the intensity scale
plt.colorbar(scatter_plot, label = 'elevation (m)')

# Add labels and title to the plot (optional)
plt.xlabel('x coordinate (m)')
plt.ylabel('Y-axis')
plt.title('Mountain y coordinate (m)')

# Show the plot
plt.show(block=True)
