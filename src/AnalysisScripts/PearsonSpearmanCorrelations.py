import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

# Load and correct PCA distances matrix
pca_distances_path = '/mnt/data/pca_points_distanceMatrix_A0A0A0MSM3.xlsx'
df_pca_matrix = pd.read_excel(pca_distances_path, header=None)

# Assuming the first row and column are indices and not part of the data
df_pca_matrix_corrected = df_pca_matrix.iloc[1:, 1:].reset_index(drop=True)

# Convert the PCA matrix to a long format DataFrame
pca_distances_long = []
for i in range(len(df_pca_matrix_corrected)):
    for j in range(i + 1, len(df_pca_matrix_corrected)):  # Adjusted range to avoid IndexError
        pca_distances_long.append({'Residue1': i + 1, 'Residue2': j + 1, 'PCADistance': df_pca_matrix_corrected.iat[i, j]})  # Adjusted to use 1-based indexing
df_pca_long = pd.DataFrame(pca_distances_long)

# Load COM distances
com_distances_path = '/mnt/data/distances_centers_of_mass_A0A0A0MSM3.xlsx'
df_com_distances = pd.read_excel(com_distances_path)

# Assuming COM distances are correct and using them to filter PCA distances
df_com_distances['PairKey'] = df_com_distances.apply(lambda x: f"{x['Residue1Chain']}{x['Residue1']}-{x['Residue2Chain']}{x['Residue2']}", axis=1)
df_pca_long['PairKey'] = df_pca_long.apply(lambda x: f"A{x['Residue1']}-A{x['Residue2']}", axis=1)  # Assuming all in chain A for simplification

# Filter PCA distances to keep only those pairs present in COM distances
df_pca_long_filtered = df_pca_long[df_pca_long['PairKey'].isin(df_com_distances['PairKey'])]

# Normalize distances
scaler = MinMaxScaler()
com_distances_scaled = scaler.fit_transform(df_com_distances['Distance'].values.reshape(-1, 1)).flatten()
pca_distances_scaled = scaler.fit_transform(df_pca_long_filtered['PCADistance'].values.reshape(-1, 1)).flatten()

# Calculate correlations and p-values
pearson_corr, pearson_p_value = pearsonr(com_distances_scaled, pca_distances_scaled)
spearman_corr, spearman_p_value = spearmanr(com_distances_scaled, pca_distances_scaled)

print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p_value:.4e}")
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p_value:.4e}")