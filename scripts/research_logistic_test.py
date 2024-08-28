import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the dataset and remove duplicates
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Initial correlation check
initial_corr_matrix = features.corr()
print("\nInitial correlation matrix with 'label':")
print(initial_corr_matrix['label'].sort_values(ascending=False))

# Remove highly correlated features
correlation_threshold = 0.8
highly_correlated_features = initial_corr_matrix.index[initial_corr_matrix['label'].abs() > correlation_threshold].tolist()
highly_correlated_features.remove('label')  # Remove the target variable itself
features = features.drop(columns=highly_correlated_features)

# Re-evaluate the correlation matrix
corr_matrix = features.corr()
print("\nUpdated correlation matrix with 'label':")
print(corr_matrix['label'].sort_values(ascending=False))

# Identify remaining highly correlated features to remove
additional_highly_correlated_features = corr_matrix.index[corr_matrix['label'].abs() > correlation_threshold].tolist()
additional_highly_correlated_features.remove('label')
print(f"Additional highly correlated features to consider removing: {additional_highly_correlated_features}")

# Remove additional highly correlated features
features_filtered = features.drop(columns=additional_highly_correlated_features)

# Ensure proper stratified split
X = features_filtered.drop(columns=['label'])
y = features_filtered['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the data
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train a simple model
simple_model = LogisticRegression(max_iter=1000)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(simple_model, X_train_resampled, y_train_resampled, cv=kf, scoring='roc_auc', n_jobs=-1)
print(f"Cross-validation scores for Logistic Regression: {cv_scores}")
print(f"Mean CV score for Logistic Regression: {cv_scores.mean():.2f}")

simple_model.fit(X_train_resampled, y_train_resampled)
y_pred = simple_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Logistic Regression Accuracy: {accuracy:.2f}')
print(f'Logistic Regression Precision: {precision:.2f}')
print(f'Logistic Regression Recall: {recall:.2f}')
print(f'Logistic Regression F1-Score: {f1:.2f}')
print(f'Logistic Regression ROC-AUC: {roc_auc:.2f}')
print('Confusion Matrix:')
print(cm)

# Save the simple model
dump(simple_model, 'logistic_regression_model_no_tfidf.joblib')
