import os
import requests
import pandas as pd
from tranco import Tranco

# Get the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def fetch_openphish_data():
    save_path = os.path.join(ROOT_DIR, 'data/raw/openphish_data.csv')
    openphish_url = "https://openphish.com/feed.txt"
    response = requests.get(openphish_url)
    urls = response.text.splitlines()

    phish_data = pd.DataFrame(urls, columns=['url'])
    phish_data['label'] = 1
    phish_data.to_csv(save_path, index=False)
    print(f"OpenPhish data saved to {save_path}")


def fetch_urlhaus_data():
    save_path = os.path.join(ROOT_DIR, 'data/raw/urlhaus_data.csv')
    urlhaus_url = "https://urlhaus.abuse.ch/downloads/text/"
    response = requests.get(urlhaus_url)
    urls = response.text.splitlines()
    urls = [url for url in urls if not url.startswith('#')]

    phish_data = pd.DataFrame(urls, columns=['url'])
    phish_data['label'] = 1
    phish_data.to_csv(save_path, index=False)
    print(f"URLhaus data saved to {save_path}")


def fetch_tranco_data():
    save_path = os.path.join(ROOT_DIR, 'data/raw/tranco_top_sites.csv')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = Tranco(cache=True, cache_dir='.tranco')
    tranco_list = t.list()
    top_sites = tranco_list.top(100000)  # Fetch the top 100,000 sites

    # Generate ranks for the domains
    ranks = list(range(1, len(top_sites) + 1))
    tranco_data = pd.DataFrame(list(zip(ranks, top_sites)), columns=['rank', 'domain'])
    tranco_data['label'] = 0
    tranco_data['url'] = 'https://' + tranco_data['domain']

    # If the file exists, ensure it can be written to
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except PermissionError as e:
            print(f"Failed to remove existing file {save_path}: {e}")
            return

    tranco_data.to_csv(save_path, index=False)
    print(f"Tranco data saved to {save_path}")


def fetch_malicious_phish_data():
    file_path = os.path.join(ROOT_DIR, 'data/raw/malicious_phish.csv')
    if not os.path.exists(file_path):
        print(f"{file_path} not found.")
        return pd.DataFrame(columns=['url', 'label'])

    phish_data = pd.read_csv(file_path)
    phish_data['label'] = phish_data['type'].apply(lambda x: 0 if x == 'benign' else 1)
    phish_data = phish_data[['url', 'label']]

    # Add "https://www." to benign URLs if they don't start with "http://" or "https://"
    def ensure_http(url, label):
        if label == 0 and not (url.startswith('http://') or url.startswith('https://')):
            return 'https://www.' + url
        return url

    phish_data['url'] = phish_data.apply(lambda row: ensure_http(row['url'], row['label']), axis=1)

    print(f"Malicious Phish data loaded from {file_path}")
    return phish_data


def merge_datasets():
    openphish_data_path = os.path.join(ROOT_DIR, 'data/raw/openphish_data.csv')
    urlhaus_data_path = os.path.join(ROOT_DIR, 'data/raw/urlhaus_data.csv')
    tranco_data_path = os.path.join(ROOT_DIR, 'data/raw/tranco_top_sites.csv')
    output_path = os.path.join(ROOT_DIR, 'data/processed/combined_url_dataset.csv')

    openphish_data = pd.read_csv(openphish_data_path)
    urlhaus_data = pd.read_csv(urlhaus_data_path)
    tranco_data = pd.read_csv(tranco_data_path)
    malicious_phish_data = fetch_malicious_phish_data()

    # Combine all malicious datasets
    malicious_data = pd.concat([openphish_data, urlhaus_data, malicious_phish_data])

    # Print counts before merging
    print(f"Number of malicious URLs before merging: {len(malicious_data)}")
    print(f"Number of benign URLs before merging: {len(tranco_data)}")

    # Combine datasets
    combined_data = pd.concat([malicious_data, tranco_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print counts after merging
    print(f"Total number of URLs after merging: {len(combined_data)}")

    # Print counts of malicious and benign URLs after merging
    num_malicious = combined_data['label'].sum()
    num_benign = len(combined_data) - num_malicious
    print(f"Number of malicious URLs after merging: {num_malicious}")
    print(f"Number of benign URLs after merging: {num_benign}")

    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT_DIR, 'data/raw'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'data/processed'), exist_ok=True)

    fetch_openphish_data()
    fetch_urlhaus_data()
    fetch_tranco_data()
    merge_datasets()
