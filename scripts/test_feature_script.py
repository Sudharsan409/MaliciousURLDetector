import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check data types
print("\nData types of each column:")
print(data.dtypes)

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Check unique values for categorical features
if 'Has_rank' in data.columns:
    print("\nUnique values in 'Has_rank':")
    print(data['Has_rank'].value_counts())

# Distribution of numerical features
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

plt.figure(figsize=(20, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Distribution of numerical features
numerical_features = ['rank', 'IP_in_URL', 'URL_len', 'Domain_len']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Distribution of a few TF-IDF features
# tfidf_features = [f'tfidf_{i}' for i in range(100)]
# for feature in tfidf_features[:5]:  # Display only the first 5 for brevity
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data[feature], kde=True)
#     plt.title(f'Distribution of {feature}')
#     plt.show()

# Manual spot check for a few rows
print("\nManual spot check:")
print(data.sample(5))
