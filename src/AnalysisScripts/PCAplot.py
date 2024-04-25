import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Assuming `data_reduced_reshaped` is your (num_res, 3) tensor from PCA reduction
# Let's simulate some data for this example
num_res = 10  # Number of residues, for example
data_reduced_reshaped = np.random.rand(num_res, 3) * 200 - 100  # Random 3D points in the range [-100, 100]

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d', facecolor='none')

# Set background color of the figure to transparent
fig.patch.set_alpha(0.0)

# Plotting each point and labeling it
for i in range(data_reduced_reshaped.shape[0]):
    ax.scatter(data_reduced_reshaped[i, 0], data_reduced_reshaped[i, 1], data_reduced_reshaped[i, 2], color='cyan')
    ax.text(data_reduced_reshaped[i, 0], data_reduced_reshaped[i, 1], data_reduced_reshaped[i, 2], '%s' % (str(i+1)), size=10, zorder=1, color='black')

# Customizing the plot
ax.grid(False)  # Disable the grid
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Setting fixed axis lengths
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

# Save the figure with transparent background
plt.savefig('3D_Plot_Publication_Fixed_Axis.png', transparent=True, dpi=300)

plt.show()