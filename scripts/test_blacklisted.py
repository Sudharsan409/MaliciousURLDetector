import logging
import os

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_phishing_urls(path):
    try:
        with open(path, 'r') as file:
            phishing_urls = set(file.read().splitlines())
        logger.info(f"Loaded {len(phishing_urls)} phishing URLs")
        return phishing_urls
    except Exception as e:
        logger.error(f"Failed to load phishing URLs from {path}: {e}")
        return set()

def is_blacklisted(url, phishing_urls):
    return 1 if url in phishing_urls else 0

# Path to phishing URL file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
phishing_url_path = os.path.join(ROOT_DIR, 'data/raw/urlhaus.txt')

try:
    logger.info("Loading phishing URLs")
    phishing_urls = load_phishing_urls(phishing_url_path)
except Exception as e:
    logger.error(f"Error loading phishing URLs: {e}")
    phishing_urls = set()  # Ensuring phishing_urls is always a set

# Example DataFrame
data = pd.DataFrame({
    'url': ['http://example1.com', 'http://malicious-url.com', 'http://example2.com','http://61.3.139.21:41078/bin.sh','http://117.255.106.126:49296/bin.sh','http://livetrack.in/EmployeeMasterImages/qace.jpg']
})

# Apply the blacklist check
data['Is_blacklisted'] = data['url'].apply(lambda x: is_blacklisted(x, phishing_urls))

print(data)
