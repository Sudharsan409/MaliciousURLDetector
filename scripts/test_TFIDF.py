import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Example DataFrame with some ranks missing
data = pd.DataFrame({
    'url': ['http://example1.com', 'http://malicious-url.com', 'http://example2.com'],
    'rank': [100, None, 50]  # Example ranks with a missing value
})


# Feature extraction functions (previously defined)
def extract_features(data):
    # Check if rank is present in the dataset
    if 'rank' in data.columns:
        max_rank = data['rank'].max()  # Find the maximum rank
        if pd.isna(max_rank):
            max_rank = 0  # If all ranks are NaN, set max_rank to 0

        # Create Has_rank column: 1 if the original rank was not NaN, else 0
        data['Has_rank'] = data['rank'].notna().astype(int)

        # Fill missing ranks with max_rank + 1
        fill_value = max_rank + 1
        data['rank'] = data['rank'].fillna(fill_value)

        # Debug: Check after filling NaN values
        print(f"Max rank: {max_rank}, Fill value: {fill_value}")
        print(data)
    else:
        # If rank is not present, add default values
        data['rank'] = -1
        data['Has_rank'] = 0

    # Add other feature extraction functions as needed
    # For example: data['url_length'] = data['url'].apply(url_length)

    return data


# Apply feature extraction
data = extract_features(data)

# Apply TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=100)
tfidf_features = tfidf.fit_transform(data['url']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

# Combine TF-IDF features with existing features
combined_data = pd.concat([data, tfidf_df], axis=1)

# Save the TF-IDF vectorizer
tfidf_path = '//models/tfidf_vectorizer.joblib'
joblib.dump(tfidf, tfidf_path)

# Final DataFrame with combined features
print(combined_data)
