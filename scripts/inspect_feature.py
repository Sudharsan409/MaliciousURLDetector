import os

import pandas as pd

# Load the extracted features
features_path = model_path = os.path.join(os.path.dirname(__file__), '..', 'data/processed', 'url_features.csv')
features_df = pd.read_csv(features_path)

# Display the first few rows of the dataframe
print(features_df.head())

# Verify the presence of the 'Has_rank' feature and filled 'rank' values
print(features_df[['rank', 'Has_rank']].describe())

# Check for any missing values
print(features_df.isnull().sum())

# Get summary statistics for each feature
print(features_df.describe())