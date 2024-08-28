import unittest
import json
from app import app


class MaliciousURLDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        payload = [
            {
                "url": "http://example.com",
                "rank": 123,
                "Dots_in_Domain": 2,
                "Https_in_URL": 1,
                "URL_len": 21,
                "Special_char_in_URL": 3,
                "Alphabets_in_URL": 14,
                "Num_unique_chars": 10,
                "Domain_len": 10,
                "Numbers_in_URL": 1,
                "Lower_case_letters_in_URL": 12
            },
            {
                "url": "http://malicious.com",
                "rank": 456,
                "Dots_in_Domain": 1,
                "Https_in_URL": 0,
                "URL_len": 17,
                "Special_char_in_URL": 1,
                "Alphabets_in_URL": 10,
                "Num_unique_chars": 8,
                "Domain_len": 7,
                "Numbers_in_URL": 2,
                "Lower_case_letters_in_URL": 8
            }
        ]
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)


if __name__ == '__main__':
    unittest.main()
