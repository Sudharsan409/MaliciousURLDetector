import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset and preprocess
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
features = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

# Ensure proper stratified split and normalization
# Splitting the dataset into 60% training and 40% testing
X = features.drop(columns=['label'])
y = features['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train a Random Forest model to get feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances')
plt.show()

print(feature_importances)

# Check for possible data leakage by analyzing the most important features
most_important_features = feature_importances.head(20)['Feature'].tolist()
print("Most important features to check for data leakage:")
print(most_important_features)

# Manually inspect these features to ensure no data leakage
print("Sample data for the most important features:")
print(features[most_important_features + ['label']].sample(10))
