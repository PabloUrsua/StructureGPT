from sklearn.decomposition import PCA
import numpy as np

# Assuming `data` is your dataset with shape (num_samples, num_residues, 114)
# where num_samples is the number of protein structures you have,
# num_residues is the number of amino acids in each protein,
# and 114 is the number of features (atomic cartesian coordinates) for each amino acid.

# Example data generation (replace this with your actual data loading process)
num_samples = 10  # Number of protein structures
num_residues = 50  # Number of amino acids per protein structure
data = np.random.rand(num_samples, num_residues, 114)  # Random data for demonstration

# Flattening the data from 3D to 2D, as PCA requires a 2D array as input
# New shape will be (num_samples*num_residues, 114)
data_flattened = data.reshape(-1, 114)

# Initialize PCA model to reduce dimensions to 3
pca = PCA(n_components=3)

# Fit the model and transform the data
data_reduced = pca.fit_transform(data_flattened)

# Reshape the data back to its original structure but with reduced dimensionality
# New shape will be (num_samples, num_residues, 3)
data_reduced_reshaped = data_reduced.reshape(num_samples, num_residues, 3)

# Now, `data_reduced_reshaped` is your dimensionality reduced data
print("Shape of the reduced data:", data_reduced_reshaped.shape)