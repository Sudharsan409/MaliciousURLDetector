import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'voting_classifier_model.joblib')
feature_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.joblib')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the feature names
with open(feature_path, 'rb') as f:
    feature_names = joblib.load(f)

print(f"Loading model from {model_path}")

# Load the trained model
try:
    voting_classifier = joblib.load(model_path)
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    exit(1)
# Get feature importances from the model (example with RandomForest)
importances = voting_classifier.estimators_[0].feature_importances_  # Assuming the first estimator is RandomForest

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(ROOT_DIR, 'feature_analysis_importances.png'))
plt.show()
