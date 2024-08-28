import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_path = os.path.join(ROOT_DIR, 'data/processed/url_features_reduced.csv')
drop_columns_path = os.path.join(ROOT_DIR, 'report/dropped_columns.txt')

data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features_cleaned.csv'))
data = data.drop(columns=['url'])

# Calculate the correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Print the names of the columns to be dropped
print("Columns to be dropped due to high correlation:", to_drop)

# Save the names of the dropped columns to a text file
with open(drop_columns_path, 'w') as f:
    for column in to_drop:
        f.write(f"{column}\n")

# Drop features
data_reduced = data.drop(columns=to_drop)

# Save the reduced dataset
data_reduced.to_csv(output_path, index=False)
print(f"Reduced data saved to {output_path} ")
print(f"List of dropped columns saved to {drop_columns_path} ")
