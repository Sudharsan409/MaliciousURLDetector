import os

import pandas as pd

# Load the dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Display some samples of benign and malicious URLs
print("Sample benign URLs:")
print(data[data['label'] == 0].head())

print("\nSample malicious URLs:")
print(data[data['label'] == 1].head())
