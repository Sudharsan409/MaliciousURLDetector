import requests
import json


GOOGLE_API_KEY = 'AIzaSyCEJCUTxTy7lBCTQJT0lH6ko56bFQGyHDY'
GOOGLE_SAFE_BROWSING_API_URL = 'https://safebrowsing.googleapis.com/v4/threatMatches:find?key=' + GOOGLE_API_KEY

def test_google_safe_browsing(api_url, api_key, test_url):
    body = {
        "client": {
            "clientId": "891669380965-rpsbaapls7eidvq25eiergo9ii1t3i4f.apps.googleusercontent.com",
            "clientVersion": "1.5.2"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [
                {"url": "http://" + test_url},
                {"url": "https://" + test_url}
            ]
        }
    }

    try:
        response = requests.post(api_url, json=body, timeout=10)
        if response.status_code == 200:
            threat_matches = response.json().get('matches', [])
            if threat_matches:
                print(f"The URL {test_url} is found to be malicious.")
            else:
                print(f"The URL {test_url} is safe.")
        else:
            print(f"Non-200 response from Google Safe Browsing API: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Exception occurred: {e}")

# Test with a known URL
test_url = "testsafebrowsing.appspot.com/s/malware.html"
test_google_safe_browsing(GOOGLE_SAFE_BROWSING_API_URL, GOOGLE_API_KEY, test_url)
