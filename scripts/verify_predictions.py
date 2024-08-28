import os
import pandas as pd
import joblib
from feature_extraction import extract_features, preprocess_for_model  # Ensure this import points to the correct module

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'voting_classifier_model.joblib')
feature_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.joblib')

print(f"Loading model from {model_path}")

# Load the trained model
try:
    voting_classifier = joblib.load(model_path)
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    exit(1)

# Define a small set of known URLs with their labels (0 for benign, 1 for malicious)
known_urls = [
    {"url": "https://www.google.com", "label": 0},  # Benign
    {"url": "http://malicious.com", "label": 1},   # Malicious
    {"url": "http://benignsite.com", "label": 0},   # Benign
    {"url": "http://phishingsite.com", "label": 1}  # Malicious
]

# Convert the known URLs to a DataFrame
known_urls_df = pd.DataFrame(known_urls)

# Extract features from the known URLs
extracted_features = extract_features(known_urls_df.copy())

# Preprocess for the model
features_for_model = preprocess_for_model(extracted_features)

# Ensure the feature order matches the model
with open(feature_path, 'rb') as f:
    feature_names = joblib.load(f)

# Align features
features_for_model = features_for_model.reindex(columns=feature_names, fill_value=0)


# Make predictions using the trained model
predictions_proba = voting_classifier.predict_proba(features_for_model)[:, 1]

# Add predictions to the DataFrame
known_urls_df['predicted_proba'] = predictions_proba
known_urls_df['predicted_label'] = (predictions_proba > 0.5).astype(int)

# Print the results
for index, row in known_urls_df.iterrows():
    print(f"URL: {row['url']}, True Label: {row['label']}, "
          f"Predicted Probability of Malicious: {row['predicted_proba']*100:.2f}%, "
          f"Predicted Label: {'Malicious' if row['predicted_label'] == 1 else 'Benign'}")

# Print the final DataFrame for verification
print(known_urls_df)
