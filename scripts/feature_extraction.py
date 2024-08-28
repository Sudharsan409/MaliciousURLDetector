import hashlib
import logging
import os
import re
import string
import time
from itertools import groupby
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tld import get_tld

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress specific urllib3 warnings
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def log_system_metrics(stage):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_usage = psutil.cpu_percent(interval=1)
    logger.info(f"{stage} - Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB, CPU usage: {cpu_usage:.2f}%")


# Get the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Load phishing URLs from local file
def load_phishing_urls(path):
    try:
        with open(path, 'r') as file:
            phishing_urls = set(file.read().splitlines())
        return phishing_urls
    except Exception as e:
        logger.error(f"Failed to load phishing URLs from {path}: {e}")
        return set()


# Define feature extraction functions

def ip_in_url(url):
    ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,7}:|:\b'
    return 1 if re.search(ipv4_pattern, url) or re.search(ipv6_pattern, url) else 0


def count_dots(url):
    return url.count('.')


def url_length(url):
    return len(url)


def count_www(url):
    return url.count('www')


def get_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return ''


def domain_length(url):
    domain = get_domain(url)
    return len(domain)


def count_hyphens(url):
    return url.count('-')


def count_underscores(url):
    return url.count('_')


def count_double_slashes(url):
    return url.count('//')


def count_at(url):
    return url.count('@')


def count_hash(url):
    return url.count('#')


def count_semicolon(url):
    return url.count(';')


def count_and(url):
    return url.count('&')


def count_http_https(url):
    url_lower = url.lower()
    count_https = url_lower.count('https')
    # Remove 'https' from the URL for accurate 'http' count
    url_lower_no_https = url_lower.replace('https', '')
    count_http = url_lower_no_https.count('http')
    return count_http, count_https


def count_numbers(url):
    return sum(c.isdigit() for c in url)


def ratio_numbers(url):
    return sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0


def count_alphabets(url):
    return sum(c.isalpha() for c in url)


def ratio_alphabets(url):
    return sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0


def count_special_chars(url):
    return sum(c in string.punctuation for c in url)


def count_other_special_chars(url):
    # Define a set of already counted special characters
    already_counted = {'.', '_', '/', '@', '#', ';', '&'}

    # Use a regex to find all special characters in the URL
    special_chars = re.findall(r'[^a-zA-Z0-9]', url)

    # Filter out the already counted special characters
    other_special_chars = [char for char in special_chars if char not in already_counted]

    return len(other_special_chars)


def ratio_special_chars(url):
    return sum(c in string.punctuation for c in url) / len(url) if len(url) > 0 else 0


def avg_word_length(url):
    words = re.findall(r'\b\w+\b', url)
    return sum(len(word) for word in words) / len(words) if words else 0


def english_words_count(url):
    english_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
                     "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                     "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
                     "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
                     "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
                     "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                     "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
                     "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
                     "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
                     "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
                     "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
                     "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
                     "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
                     "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
                     "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
                     "your", "yours", "yourself", "yourselves"}
    words = re.findall(r'\b\w+\b', url)
    return sum(1 for word in words if word.lower() in english_words)


def avg_english_word_length(url):
    words = re.findall(r'\b\w+\b', url)
    english_words = [word for word in words if re.match(r'[a-zA-Z]+', word)]
    return sum(len(word) for word in english_words) / len(english_words) if english_words else 0


def path_length(url):
    try:
        path = urlparse(url).path
        return len(path)
    except Exception:
        return 0


def query_length(url):
    try:
        query = urlparse(url).query
        return len(query)
    except Exception:
        return 0


def count_params(url):
    query = urlparse(url).query
    if not query:
        return 0
    return len(parse_qs(query))


def count_unique_chars(url):
    return len(set(url))


def max_consecutive_chars(url):
    return max(len(list(group)) for _, group in groupby(url))


def tld_length(tld):
    try:
        return len(tld)
    except TypeError:
        return -1


def subdomain_length(url):
    subdomain = re.findall(r'://(?:www\.)?([^/:]+)', url)
    parts = subdomain[0].split('.') if subdomain else []
    return len(parts[0]) if len(parts) > 2 else 0


def contains_suspicious_keywords(url):
    keywords = ["login", "buy", "free", "click", "sale", "sign", "verify", "update", "account", "ebayisapi",
                "webscr", "pay"]
    return 1 if any(keyword in url for keyword in keywords) else 0


# Shortened URL detection
shortening_services = (r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog'
                       r'\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to'
                       r'|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com'
                       r'|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.ly|s\.gd|qr\.ae|adf\.ly|bitly'
                       r'\.com|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u'
                       r'\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|v\.gd|vurl'
                       r'\.bz|tinylink\.in|all\.ly|lnkd\.in|dld\.bz|aka\.gr|adfoc\.us|link\.zip\.net')


def is_shortened_url(url):
    return 1 if re.search(shortening_services, url) else 0


def is_blacklisted(url, phishing_urls):
    return 1 if url in phishing_urls else 0


def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()


# Extract features
def extract_features():
    scaler = MinMaxScaler()
    input_path = os.path.join(ROOT_DIR, 'data/processed/preprocessed_url_dataset.csv')
    output_path = os.path.join(ROOT_DIR, 'data/processed/url_features.csv')
    phishing_url_path = os.path.join(ROOT_DIR, 'data/raw/urlhaus.txt')

    logger.info(f"Reading input data from {input_path}")
    start_time = time.time()
    try:
        data = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Error reading input data: {e}")
        return None

    logger.info(f"Data read in {time.time() - start_time:.2f} seconds")

    # Lexical feature extraction
    try:
        start_time = time.time()
        logger.info("Extracting lexical features")
        log_system_metrics("Before lexical feature extraction")

        data['IP_in_URL'] = data['url'].apply(ip_in_url)
        data['URL_len'] = data['url'].apply(url_length)
        data['Domain_len'] = data['url'].apply(domain_length)

        data['Hyphens_in_Domain'] = data['url'].apply(count_hyphens)
        data['Underscores_in_Domain'] = data['url'].apply(count_underscores)
        data['Double_slashes_in_URL'] = data['url'].apply(count_double_slashes)
        data['At(@)_in_URL'] = data['url'].apply(count_at)
        data['Hash(#)_in_URL'] = data['url'].apply(count_hash)
        data['Semicolon(;)_in_URL'] = data['url'].apply(count_semicolon)
        data['And(&)_in_URL'] = data['url'].apply(count_and)
        data['Other_special_chars_in_URL'] = data['url'].apply(count_other_special_chars)
        data['Special_char_ratio_in_URL'] = data['url'].apply(ratio_special_chars)
        data['Dots_in_Domain'] = data['url'].apply(count_dots)
        data['Normalized_Dots_in_Domain'] = scaler.fit_transform(data[['Dots_in_Domain']])
        # Try lowering the threshold to see if it balances the feature better
        data['Dots_in_Domain_Binned'] = pd.cut(data['Dots_in_Domain'], bins=[-1, 0, 1, 2, 3, np.inf],
                                               labels=['0', '1', '2', '3', '4'])

        # Check the distribution of the new binned feature
        print(data['Dots_in_Domain_Binned'].value_counts())

        data['Dots_DomainLen_Interaction'] = data['Dots_in_Domain'] * data['Domain_len']
        data['Dots_UrlLen_Interaction'] = data['Dots_in_Domain'] * data['URL_len']

        # Check the new interaction terms
        print(data[['Dots_DomainLen_Interaction', 'Dots_UrlLen_Interaction']].head())

        data['Dots_Hyphens_Combination'] = data['Dots_in_Domain'] + data['Hyphens_in_Domain']

        # Check the new composite feature
        print(data['Dots_Hyphens_Combination'].value_counts())

        scaler = StandardScaler()
        data[['URL_len', 'Domain_len', 'Dots_in_Domain', 'Hyphens_in_Domain']] = scaler.fit_transform(
            data[['URL_len', 'Domain_len', 'Dots_in_Domain', 'Hyphens_in_Domain']])

        # Verify the scaling
        print(data[['URL_len', 'Domain_len', 'Dots_in_Domain', 'Hyphens_in_Domain']].head())

        data[['Http_in_URL', 'Https_in_URL']] = pd.DataFrame(data['url'].apply(count_http_https).tolist(),
                                                             index=data.index)

        data['Numbers_in_URL'] = data['url'].apply(count_numbers)
        data['Numbers_ratio_in_URL'] = data['url'].apply(ratio_numbers)
        data['Alphabets_in_URL'] = data['url'].apply(count_alphabets)
        data['Alphabet_ratio_in_URL'] = data['url'].apply(ratio_alphabets)
        data['Num_unique_chars'] = data['url'].apply(count_unique_chars)
        data['Max_consecutive_chars'] = data['url'].apply(max_consecutive_chars)
        data['English_words_in_URL'] = data['url'].apply(english_words_count)
        data['Avg_english_word_len_in_URL'] = data['url'].apply(avg_english_word_length)
        data['Avg_word_len_in_URL'] = data['url'].apply(avg_word_length)
        data['Contains_suspicious_keywords'] = data['url'].apply(contains_suspicious_keywords)

        data['Is_shortened_URL'] = data['url'].apply(is_shortened_url)

        data['Path_len'] = data['url'].apply(path_length)
        data['Query_len'] = data['url'].apply(query_length)
        data['Num_params'] = data['url'].apply(count_params)
        data['tld'] = data['url'].apply(lambda i: get_tld(i, fail_silently=True))
        data['TLD_len'] = data['url'].apply(tld_length)
        data['Subdomain_len'] = data['url'].apply(subdomain_length)

        # Call interaction term function
        data = create_interaction_terms(data)

        logger.info(f"Lexical features extracted in {time.time() - start_time:.2f} seconds")
        log_system_metrics("After lexical feature extraction")
    except Exception as e:
        logger.error(f"Error extracting lexical features: {e}")
        return None

    # Load phishing URLs
    try:
        logger.info("Loading phishing URLs")
        phishing_urls = load_phishing_urls(phishing_url_path)
        logger.info(f"Loaded {len(phishing_urls)} phishing URLs")
    except Exception as e:
        logger.error(f"Error loading phishing URLs: {e}")
        return None

    # Extract domain from URL for DNS-based features
    try:
        logger.info("Extracting DNS-based features")
        start_time = time.time()
        log_system_metrics("Before DNS-based feature extraction")
        data['domain'] = data['url'].apply(
            lambda x: re.findall(r'://(?:www\.)?([^/:]+)', x)[0] if re.findall(r'://(?:www\.)?([^/:]+)', x) else '')
        # data['Is_blacklisted'] = data['url'].apply(lambda x: is_blacklisted(x, phishing_urls))
        logger.info(f"DNS-based features extracted in {time.time() - start_time:.2f} seconds")
        log_system_metrics("After DNS-based feature extraction")
    except Exception as e:
        logger.error(f"Error extracting DNS-based features: {e}")
        return None

    # Check if rank is present in the dataset
    # if 'rank' in data.columns:
    #     max_rank = data['rank'].max()  # Find the maximum rank
    #     if pd.isna(max_rank):
    #         max_rank = 0  # If all ranks are NaN, set max_rank to 0
    #
    #     # Create Has_rank column: 1 if the original rank was not NaN, else 0
    #     #data['Has_rank'] = data['rank'].notna().astype(int)
    #
    #     # Fill missing ranks with max_rank + 1
    #     fill_value = max_rank + 1
    #     data['rank'] = data['rank'].fillna(fill_value)
    #
    # else:
    #     # If rank is not present, add default values
    #     data['rank'] = -1
    #     #data['Has_rank'] = 0

    try:
        logger.info("Combining all features")
        start_time = time.time()
        data.drop(columns=['domain', 'tld'], inplace=True)
        # data.drop(columns=['domain', 'url'], inplace=True)  # Drop intermediate domain and URL columns
        data.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error combining features: {e}")
        return None

    return data


def create_interaction_terms(data):
    # Select features for interaction
    features_to_interact = [
        ('URL_len', 'Domain_len'),
        ('URL_len', 'Contains_suspicious_keywords'),
        ('Domain_len', 'Contains_suspicious_keywords')
    ]

    # Create interaction terms
    for feature1, feature2 in features_to_interact:
        interaction_term_name = f"{feature1}_{feature2}_interaction"
        data[interaction_term_name] = data[feature1] * data[feature2]
        print(f"Created interaction term: {interaction_term_name}")

    return data


# Define the extract_features function
def extract_features_url(url):
    scaler = StandardScaler()
    features = {}
    features['IP_in_URL'] = ip_in_url(url)
    features['URL_len'] = url_length(url)
    features['Domain_len'] = domain_length(url)
    features['Hyphens_in_Domain'] = count_hyphens(url)
    features['Underscores_in_Domain'] = count_underscores(url)
    features['Double_slashes_in_URL'] = count_double_slashes(url)
    features['At(@)_in_URL'] = count_at(url)
    features['Hash(#)_in_URL'] = count_hash(url)
    features['Semicolon(;)_in_URL'] = count_semicolon(url)
    features['And(&)_in_URL'] = count_and(url)
    features['Other_special_chars_in_URL'] = count_other_special_chars(url)
    features['Special_char_ratio_in_URL'] = ratio_special_chars(url)
    features['Dots_in_Domain'] = count_dots(url)
    features['Dots_in_Domain_Binned'] = \
        pd.cut([features['Dots_in_Domain']], bins=[-1, 0, 1, 2, 3, np.inf], labels=['0', '1', '2', '3', '4'])[0]

    features['Dots_DomainLen_Interaction'] = features['Dots_in_Domain'] * features['Domain_len']
    features['Dots_UrlLen_Interaction'] = features['Dots_in_Domain'] * features['URL_len']
    features['Dots_Hyphens_Combination'] = features['Dots_in_Domain'] + features['Hyphens_in_Domain']

    features['Http_in_URL'], features['Https_in_URL'] = count_http_https(url)
    features['Numbers_in_URL'] = count_numbers(url)
    features['Numbers_ratio_in_URL'] = ratio_numbers(url)
    # features['Alphabets_in_URL'] = count_alphabets(url)
    # features['Alphabet_ratio_in_URL'] = ratio_alphabets(url)
    features['Num_unique_chars'] = count_unique_chars(url)
    features['Max_consecutive_chars'] = max_consecutive_chars(url)
    features['English_words_in_URL'] = english_words_count(url)
    features['Avg_english_word_len_in_URL'] = avg_english_word_length(url)
    # features['Avg_word_len_in_URL'] = avg_word_length(url)
    features['Contains_suspicious_keywords'] = contains_suspicious_keywords(url)
    features['Is_shortened_URL'] = is_shortened_url(url)
    features['Path_len'] = path_length(url)
    features['Query_len'] = query_length(url)
    # features['Num_params'] = count_params(url)
    # features['tld'] = get_tld(url, fail_silently=True)
    # features['TLD_len'] = tld_length(url)
    features['Subdomain_len'] = subdomain_length(url)

    # Convert the features dictionary to a DataFrame
    features_df = pd.DataFrame([features])

    # Apply normalization/scaling if necessary
    # Assuming scaler has been previously defined and fit
    # scaler = MinMaxScaler()  # or StandardScaler()
    # features_df = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)

    # Create interaction terms
    features_df = create_interaction_terms(features_df)

    return features_df


if __name__ == "__main__":
    features = extract_features()
