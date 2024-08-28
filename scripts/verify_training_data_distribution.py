import os

import pandas as pd
# Plot the distribution for better visualization
import matplotlib.pyplot as plt

# Load your dataset
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

output_plot_path = os.path.join(ROOT_DIR, 'plots/training_data_distribution.png')
data = data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features_cleaned.csv'))

# Check the distribution of the 'label' column
label_distribution = data['label'].value_counts()
print("Label distribution in the dataset:")
print(label_distribution)



plt.figure(figsize=(6, 4))
label_distribution.plot(kind='bar')
plt.title('Distribution of Labels in the Dataset')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plot_path = output_plot_path
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()
