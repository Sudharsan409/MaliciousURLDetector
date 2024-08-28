import joblib
import lime
import lime.lime_tabular
import pandas as pd
from joblib import load

from app import extract_features

# Load the model
models_dir = '//models'
model = joblib.load(f'{models_dir}/voting_classifier_model.joblib')

# Extract features from a sample URL
url = "https://www.google.com"
features_df = extract_features(url)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=features_df.values,  # You should use your training data here
    feature_names=features_df.columns,
    class_names=['benign', 'malicious'],
    mode='classification'
)

# Explain the prediction for the sample URL
exp = explainer.explain_instance(
    data_row=features_df.values[0],
    predict_fn=model.predict_proba
)

# Show explanation
exp.show_in_notebook(show_all=False)
