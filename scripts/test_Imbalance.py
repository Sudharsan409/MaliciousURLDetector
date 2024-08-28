import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the extracted features
data_path = os.path.join(os.path.dirname(__file__), '..', 'data/processed', 'url_features.csv')
features = pd.read_csv(data_path)

# Check the distribution of labels
label_distribution = features['label'].value_counts()
print(label_distribution)

# Visualize the distribution
label_distribution.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
