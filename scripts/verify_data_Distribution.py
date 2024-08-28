import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Test")
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/url_features.csv'))
print("Test")
print(data['High_Dots_in_Domain'].value_counts())
sns.countplot(data['High_Dots_in_Domain'])
plt.title('Distribution of High_Dots_in_Domain')
plt.show()

# Check value counts
print(data['High_Dots_in_Domain'].value_counts())
