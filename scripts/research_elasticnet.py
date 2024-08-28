import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the dataset and remove duplicates
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv')).drop_duplicates()

# Drop columns that are known to leak information or are too strong predictors
columns_to_drop = ['Is_blacklisted', 'Dots_in_Domain']
features_filtered = features.drop(columns=[col for col in columns_to_drop if col in features.columns])

# Identify and remove highly correlated features
initial_corr_matrix = features_filtered.corr()
high_correlation_threshold = 0.5
highly_correlated_features = initial_corr_matrix.index[
    initial_corr_matrix['label'].abs() > high_correlation_threshold].tolist()
highly_correlated_features.remove('label')  # Remove the target variable itself
features_filtered = features_filtered.drop(columns=highly_correlated_features)

# Transform continuous features using KBinsDiscretizer to reduce their impact
continuous_features = ['Path_len', 'URL_len', 'Numbers_in_URL', 'Numbers_ratio_in_URL']
transformer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
features_filtered[continuous_features] = transformer.fit_transform(features_filtered[continuous_features])

# Check correlation matrix again
updated_corr_matrix = features_filtered.corr()
print("\nUpdated correlation matrix with 'label':")
print(updated_corr_matrix['label'].sort_values(ascending=False))

# Split dataset into features and target variable
X = features_filtered.drop(columns=['label'])
y = features_filtered['label']

# Split into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train a Logistic Regression model with Elastic Net regularization
logistic_model = LogisticRegression(max_iter=5000, penalty='elasticnet', l1_ratio=0.5, solver='saga')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(logistic_model, X_train_resampled, y_train_resampled, cv=kf, scoring='roc_auc', n_jobs=-1)
print(f"Cross-validation scores for Logistic Regression with Elastic Net: {cv_scores}")
print(f"Mean CV score for Logistic Regression with Elastic Net: {cv_scores.mean():.2f}")

# Train and evaluate the model on the test set
logistic_model.fit(X_train_resampled, y_train_resampled)
y_pred = logistic_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Logistic Regression with Elastic Net Accuracy: {accuracy:.2f}')
print(f'Logistic Regression with Elastic Net Precision: {precision:.2f}')
print(f'Logistic Regression with Elastic Net Recall: {recall:.2f}')
print(f'Logistic Regression with Elastic Net F1-Score: {f1:.2f}')
print(f'Logistic Regression with Elastic Net ROC-AUC: {roc_auc:.2f}')
print('Confusion Matrix:')
print(cm)

# Save the trained model
dump(logistic_model, 'logistic_regression_elastic_net_model.joblib')

# Assuming the logistic_model is the trained Logistic Regression model with Elastic Net regularization

# Get the feature names
feature_names = X.columns

# Get the coefficients of the trained logistic regression model
coefficients = logistic_model.coef_[0]

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute Coefficient': np.abs(coefficients)
}).sort_values(by='Absolute Coefficient', ascending=False)

# Display the feature importance
print("Feature Importance:")
print(feature_importance_df)

# Save the feature importance to a CSV file
feature_importance_df.to_csv('feature_importance.csv', index=False)
