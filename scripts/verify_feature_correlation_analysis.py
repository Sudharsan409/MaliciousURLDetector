import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features_cleaned.csv'))
output_plot_path = os.path.join(ROOT_DIR, 'plots/feature_correlation_matrix.png')

# Calculate the correlation matrix
data = data.drop(columns=['url'])
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plot_path = output_plot_path
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()
