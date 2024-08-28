import numpy as np
import pandas as pd

from scripts.research_elasticnet import logistic_model

# Get the feature names
feature_names = X.columns

# Get the coefficients of the trained logistic regression model
coefficients = logistic_model.coef_[0]

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)

# Save the feature importance to a CSV file
feature_importance_df.to_csv('feature_importance.csv', index=False)
