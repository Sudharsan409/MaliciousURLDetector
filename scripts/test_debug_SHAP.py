import os

import joblib
import shap
import pandas as pd
from joblib import load

from app import extract_features

# Load the model
models_dir = '//models'
model_path = os.path.join('models', 'voting_classifier_model.joblib')
model = joblib.load(f'{models_dir}/voting_classifier_model.joblib')

# Extract features from a sample URL
url = "https://www.google.com"
features_df = extract_features(url)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_df)

# Plot SHAP values for the sample URL
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], features_df)

# For global explanation
shap.summary_plot(shap_values, features_df)
