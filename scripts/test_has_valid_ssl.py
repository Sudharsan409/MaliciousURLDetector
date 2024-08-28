import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, SSLError, Timeout, ConnectionError
from urllib3.exceptions import HeaderParsingError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def has_valid_ssl(domain_name):
    session = requests.Session()
    retries = Retry(
        total=1,  # Reduced the total number of retries
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    try:
        response = session.get(f"https://{domain_name}", timeout=5)
        return 1 if response.status_code == 200 else 0
    except (RequestException, SSLError, Timeout, ConnectionError, HeaderParsingError) as e:
        logger.warning(f"Exception for {domain_name}: {e.__class__.__name__}: {e}")
        return 0


# Test the has_valid_ssl function
def test_has_valid_ssl():
    test_domains = [
        "google.com",  # Valid SSL
        "expired.badssl.com",  # Expired SSL
        "self-signed.badssl.com",  # Self-signed SSL
        "untrusted-root.badssl.com",  # Untrusted root SSL
        "wrong.host.badssl.com",  # Wrong host SSL
        "example.com"  # Valid SSL
    ]

    for domain in test_domains:
        result = has_valid_ssl(domain)
        print(f"Domain: {domain}, Has Valid SSL: {result}")


if __name__ == "__main__":
    test_has_valid_ssl()
