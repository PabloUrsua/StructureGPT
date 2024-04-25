import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data (replace with your actual results)
results = {
    'Analysis': ['Normal PCA Points', 'PCA w/ Atomic Classes Masking', 'PCA w/ Coordinates Masking'],
    'Pearson Correlation': [0.8493, 0.6626, 0.0465],
    'Spearman Correlation': [0.8258, 0.6375, 0.0070],
    'Pearson P-value': [0.0000e00, 0.0000e00, 4.5087e-07],
    'Spearman P-value': [0.000e00, 0.0000e00, 4.4666e-01]
}

df = pd.DataFrame(results)

# Increase font size
plt.rcParams.update({'font.size': 14})

# Set the style
sns.set(style="white")

# Define colors
colors = ['salmon', 'cyan']

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plotting
bar_pearson = sns.barplot(x='Analysis', y='Pearson Correlation', data=df, color=colors[0], label='Pearson')
bar_spearman = sns.barplot(x='Analysis', y='Spearman Correlation', data=df, color=colors[1], label='Spearman', alpha=0.6)

# Adjust the y-axis limit
ax.set_ylim(0, df[['Pearson Correlation', 'Spearman Correlation']].values.max() + 0.2)

# Titles and labels
plt.title('Correlation Coefficients of PCA Points Analysis')
plt.ylabel('Correlation Coefficient')
plt.legend()

# Annotating p-values
for i, row in df.iterrows():
    pearson_asterisks = '***' if row['Pearson P-value'] < 0.01 else ''
    spearman_asterisks = '***' if row['Spearman P-value'] < 0.01 else '**' if row['Spearman P-value'] < 0.05 else ''
    
    plt.text(i, row['Pearson Correlation'] + 0.01, pearson_asterisks, ha='center', color='black', fontsize=14)
    plt.text(i, row['Spearman Correlation'] + 0.01, spearman_asterisks, ha='center', color='black', fontsize=14)

# Save the figure with high dpi
plt.savefig('/mnt/data/correlation_coefficients_analysis_with_asterisks.png', dpi=300)

plt.show()