import pandas as pd
from sklearn.impute import SimpleImputer
import os

# Load the data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Identify missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# Remove columns with more than 50% missing values
threshold = 0.5
data = data.loc[:, data.isnull().mean() < threshold]

# Separate numeric and categorical features
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Impute missing values
# Impute numerical features with the mean
num_imputer = SimpleImputer(strategy='mean')
data[numeric_features] = num_imputer.fit_transform(data[numeric_features])

# Impute categorical features with the mode
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

# Verify no missing values remain
missing_values_after_imputation = data.isnull().sum()
print("Missing values after imputation:")
print(missing_values_after_imputation[missing_values_after_imputation > 0])

# Save the cleaned data

output_path = os.path.join(ROOT_DIR, 'data/processed/url_features_cleaned.csv')
data.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
