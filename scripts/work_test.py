import string
from urllib.parse import urlparse, parse_qs


def count_http_https(url):
    url_lower = url.lower()
    count_https = url_lower.count('https')
    # Remove 'https' from the URL for accurate 'http' count
    url_lower_no_https = url_lower.replace('https', '')
    count_http = url_lower_no_https.count('http')
    return count_http, count_https


def count_params(url):
    query = urlparse(url).query
    if not query:
        return 0
    return len(parse_qs(query))


# Example usage
url = "https://www.example.com"
count_http, count_https = count_http_https(url)
print(f"HTTP count: {count_http}, HTTPS count: {count_https}")
count_params = count_params(url)
print(f"Params count: {count_params}")

# Example validation
url_sample = "http://example123.com"
special_char_ratio = sum(c in string.punctuation for c in url_sample) / len(url_sample)
numbers_ratio = sum(c.isdigit() for c in url_sample) / len(url_sample)
alphabet_ratio = sum(c.isalpha() for c in url_sample) / len(url_sample)

print(f"Special Char Ratio: {special_char_ratio}")
print(f"Numbers Ratio: {numbers_ratio}")
print(f"Alphabet Ratio: {alphabet_ratio}")
