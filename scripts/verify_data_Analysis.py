import os

import pandas as pd

# Load the dataset
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Display the first few rows of the dataset and summary statistics
print(data.head(), data.describe())

missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]

missing_values_summary = missing_values.to_frame(name='Missing Values')
missing_values_summary['Percentage'] = (missing_values_summary['Missing Values'] / len(data)) * 100

print(missing_values_summary)
