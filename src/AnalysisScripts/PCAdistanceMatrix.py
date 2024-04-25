import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

# Assuming `points` is your numpy array of shape (158, 2)
# Example: points = np.random.rand(158, 2)  # Replace or use your actual data

# Calculate the distance matrix
dist_matrix = distance_matrix(points, points)

# Convert the distance matrix to a pandas DataFrame
dist_df = pd.DataFrame(dist_matrix)

# Save the DataFrame to an Excel file
dist_df.to_excel('distance_matrix.xlsx', index=False)