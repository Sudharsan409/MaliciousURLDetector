import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))

correlation_matrix = data.corr(data.drop(columns=['url']))
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
# Prepare the data
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target variable

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
