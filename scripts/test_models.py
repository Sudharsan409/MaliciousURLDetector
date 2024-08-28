import pandas as pd
from joblib import load

# Load the models
models = {
    "Random Forest": load('//models/random_forest_model.joblib'),
    "Decision Tree": load('//models/decision_tree_model.joblib'),
    "Linear SVM": load('//models/linear_svm_model.joblib'),
    "Gradient Boosting": load('//models/gradient_boosting_model.joblib'),
    "KNN": load('//models/knn_model.joblib'),
    "XGBoost": load('//models/xgboost_model.joblib'),
    "LightGBM": load('//models/lightgbm_model.joblib'),
    "Voting Classifier": load('//models/voting_classifier_model.joblib')
}

# Define the full set of features used during training
full_feature_columns = [
    'Path_len', 'URL_len', 'Numbers_in_URL', 'Dots_in_Domain', 'Domain_len',
    'Alphabets_in_URL', 'Subdomain_len', 'Num_unique_chars', 'Max_consecutive_chars', 'Hyphens_in_Domain',
    'And(&)_in_URL', 'At(@)_in_URL', 'Contains_suspicious_keywords', 'Double_slashes_in_URL', 'English_words_in_URL',
    'Hash(#)_in_URL', 'Is_shortened_URL', 'Query_len', 'Semicolon(;)_in_URL', 'Special_char_ratio_in_URL','Http_in_URL','Underscores_in_Domain','abnormal_url'
]


sample_test_data = pd.DataFrame({
    'url': ['google.com', 'malicious-example.com', 'benign-example.org'],
    'Path_len': [0, 5, 2],
    'URL_len': [10, 22, 18],
    'Numbers_in_URL': [0, 1, 0],
    'Dots_in_Domain': [1, 2, 1],
    'Domain_len': [6, 17, 13],
    'Alphabets_in_URL': [10, 20, 16],
    'Subdomain_len': [0, 3, 2],
    'Num_unique_chars': [5, 12, 8],
    'Max_consecutive_chars': [2, 3, 2],
    'Hyphens_in_Domain': [0, 1, 0],
    'abnormal_url': [0, 1, 0],
    'Underscores_in_Domain': [0, 0, 0],
    'Double_slashes_in_URL': [0, 1, 0],
    'At(@)_in_URL': [0, 0, 0],
    'Hash(#)_in_URL': [0, 0, 0],
    'Semicolon(;)_in_URL': [0, 0, 0],
    'And(&)_in_URL': [0, 0, 0],
    'Http_in_URL': [0, 0, 0],
    'Special_char_ratio_in_URL': [0.0, 0.091, 0.0],
    'English_words_in_URL': [1, 0, 1],
    'Contains_suspicious_keywords': [0, 1, 0],
    'Is_shortened_URL': [0, 0, 0],
    'Query_len': [0, 0, 0],
    'Is_shortened_URL': [1, 1, 1]
})

# Add missing feature columns with default values
for feature in full_feature_columns:
    if feature not in sample_test_data.columns:
        sample_test_data[feature] = 0  # or some default value

X_sample_test = sample_test_data[full_feature_columns]

# Print out predictions for each model
for model_name, model in models.items():
    try:
        predictions = model.predict(X_sample_test)
        print(f"{model_name} predictions: {predictions}")
    except Exception as e:
        print(f"Error in {model_name} prediction: {e}")
