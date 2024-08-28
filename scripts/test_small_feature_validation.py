import os

import pandas as pd
from feature_extraction import extract_features

# Create a small sample DataFrame for testing
test_data = pd.DataFrame({
    'url': [
        'http://google.com',
        'http://malicious.com',
        'http://benignsite.com',
        'http://phishingsite.com',
        'http://117.219.134.134:41486/Mozi.m',
        'http://dl.1003b.56a.com/pub/1003b/Patch/Patch_Data/Patch_0.3300/1003b.exe'
    ]
})

model_path = os.path.join(os.path.dirname(__file__), '..', 'data/processed', 'preprocessed_url_dataset.csv')

# Save the test data to the appropriate path
test_data.to_csv(model_path, index=False)

# Run feature extraction
features = extract_features()

# Print the extracted features
if features is not None:
    print(features.head())
else:
    print("Feature extraction failed.")
