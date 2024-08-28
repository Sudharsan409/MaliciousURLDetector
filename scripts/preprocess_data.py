# scripts/preprocess_data.py

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def preprocess_data():
    input_path = os.path.join(ROOT_DIR, 'data/processed/combined_url_dataset.csv')
    output_path = os.path.join(ROOT_DIR, 'data/processed/preprocessed_url_dataset.csv')

    try:
        data = pd.read_csv(input_path)
        logger.info(f"Data loaded successfully from {input_path}")
    except Exception as e:
        logger.error(f"Error reading input data: {e}")
        return

    initial_count = data.shape[0]

    # Remove duplicates
    data = data.drop_duplicates(subset=['url'])
    logger.info(f"Removed duplicates. {initial_count - data.shape[0]} duplicates dropped.")

    # Removing Rank
    data.drop(columns=['rank'], inplace=True)
    logger.info("Rank Column removed.")

    # Normalize URLs
    data['url'] = data['url'].str.lower()
    logger.info("URLs normalized to lowercase.")

    # Optionally: Check for missing values and handle them if necessary
    missing_values = data['url'].isnull().sum()
    if missing_values > 0:
        data = data.dropna(subset=['url'])
        logger.info(f"Removed {missing_values} rows with missing URLs.")

    # Save preprocessed data
    try:
        data.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")


if __name__ == "__main__":
    preprocess_data()
