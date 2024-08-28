import os

import numpy as np
from flask import request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from app import app
from scripts.feature_extraction import extract_features_url, create_interaction_terms

# Load your trained model
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
random_forest_model = joblib.load(f'{models_dir}/random_forest_model.joblib')
decision_tree_model = joblib.load(f'{models_dir}/decision_tree_model.joblib')
linear_svm_model = joblib.load(f'{models_dir}/linear_svm_model.joblib')
gradient_boosting_model = joblib.load(f'{models_dir}/gradient_boosting_model.joblib')
knn_model = joblib.load(f'{models_dir}/knn_model.joblib')
xgboost_model = joblib.load(f'{models_dir}/xgboost_model.joblib')
lightgbm_model = joblib.load(f'{models_dir}/lightgbm_model.joblib')
voting_classifier_model = joblib.load(f'{models_dir}/voting_classifier_model.joblib')

models = {
    'Random Forest': random_forest_model,
    'Decision Tree': decision_tree_model,
    'Linear SVM': linear_svm_model,
    'Gradient Boosting': gradient_boosting_model,
    'KNN': knn_model,
    'XGBoost': xgboost_model,
    'LightGBM': lightgbm_model,
    'Voting Classifier': voting_classifier_model
}


def preprocess_url(url):
    # Add preprocessing steps here (e.g., feature extraction)
    features_df = extract_features_url(url)  # Implement this function based on your feature extraction logic
    # Convert categorical features to numeric or category type
    features_df['Dots_in_Domain_Binned'] = features_df['Dots_in_Domain_Binned'].astype('category')
    # Ensure all features are numeric
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    # Check for any non-numeric columns
    non_numeric_columns = features_df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) > 0:
        raise ValueError(f"Non-numeric columns found: {non_numeric_columns}")

    imputer = SimpleImputer(strategy='mean')
    features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

    return features_df

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        features_df = preprocess_url(url)

        results = {}
        for model_name, model in models.items():
            prediction = model.predict(features_df)[0]

            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features_df)[0][1]
            elif model_name == 'XGBoost':
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features_df, enable_categorical=True)
                prediction_proba = model.predict(dmatrix)[0]
            else:
                # Handle models without predict_proba
                if hasattr(model, 'decision_function'):
                    decision_score = model.decision_function(features_df)
                    prediction_proba = 1 / (1 + np.exp(-decision_score))[0]
                else:
                    prediction_proba = float('nan')  # Or handle it differently

            results[model_name] = {
                'prediction': 'malicious' if prediction == 1 else 'benign',
                'probability': prediction_proba * 100
            }

        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        # Return an error response
        return jsonify({'error': str(e)}), 500


