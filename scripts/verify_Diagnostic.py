import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Check for duplicate rows
duplicates = features.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check correlation matrix
corr_matrix = features.corr()
print("\nCorrelation matrix with 'label':")
print(corr_matrix['label'].sort_values(ascending=False))

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Check distribution of the target variable
print("\nDistribution of the target variable:")
print(features['label'].value_counts())

# Check feature statistics
print("\nDescriptive statistics of the features:")
print(features.describe())

# Sample of the dataset to manually inspect
print("\nSample rows from the dataset:")
print(features.sample(10))
