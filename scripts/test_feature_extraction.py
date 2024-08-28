import unittest
import pandas as pd
from feature_extraction import (
    ip_in_url, url_length, domain_length, count_dots, count_hyphens, count_underscores, count_double_slashes, count_at,
    count_hash, count_semicolon, count_and, count_http, count_https, count_numbers, ratio_numbers, count_alphabets,
    ratio_alphabets, count_lowercase, ratio_lowercase, count_uppercase, ratio_uppercase, count_special_chars,
    ratio_special_chars, avg_word_length, english_words_count, avg_english_word_length, path_length, query_length,
    count_params, count_unique_chars, max_consecutive_chars, tld_length, subdomain_length, contains_suspicious_keywords,
    is_shortened_url, domain_age, is_blacklisted
)

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.urls = [
            {"url": "http://google.com"},
            {"url": "http://malicious.com"},
            {"url": "http://117.219.134.134:41486/Mozi.m"},
            {"url": "http://benignsite.com"},
            {"url": "http://secure-login.com"},
            {"url": "http://account-update.com"},
            {"url": "http://freebankverify.com"}
        ]

    def test_feature_extraction(self):
        features_df = pd.DataFrame(self.urls)
        features_df['IP_in_URL'] = features_df['url'].apply(ip_in_url)
        features_df['URL_len'] = features_df['url'].apply(url_length)
        features_df['Domain_len'] = features_df['url'].apply(domain_length)
        features_df['Dots_in_Domain'] = features_df['url'].apply(count_dots)
        features_df['Hyphens_in_Domain'] = features_df['url'].apply(count_hyphens)
        features_df['Underscores_in_Domain'] = features_df['url'].apply(count_underscores)
        features_df['Double_slashes_in_URL'] = features_df['url'].apply(count_double_slashes)
        features_df['At(@)_in_URL'] = features_df['url'].apply(count_at)
        features_df['Hash(#)_in_URL'] = features_df['url'].apply(count_hash)
        features_df['Semicolon(;)_in_URL'] = features_df['url'].apply(count_semicolon)
        features_df['And(&)_in_URL'] = features_df['url'].apply(count_and)
        features_df['Http_in_URL'] = features_df['url'].apply(count_http)
        features_df['Https_in_URL'] = features_df['url'].apply(count_https)
        features_df['Numbers_in_URL'] = features_df['url'].apply(count_numbers)
        features_df['Numbers_ratio_in_URL'] = features_df['url'].apply(ratio_numbers)
        features_df['Alphabets_in_URL'] = features_df['url'].apply(count_alphabets)
        features_df['Alphabet_ratio_in_URL'] = features_df['url'].apply(ratio_alphabets)
        features_df['Lower_case_letters_in_URL'] = features_df['url'].apply(count_lowercase)
        features_df['Lower_case_letters_ratio_in_URL'] = features_df['url'].apply(ratio_lowercase)
        features_df['Upper_case_letters_in_URL'] = features_df['url'].apply(count_uppercase)
        features_df['Upper_case_letters_ratio_in_URL'] = features_df['url'].apply(ratio_uppercase)
        features_df['Special_char_in_URL'] = features_df['url'].apply(count_special_chars)
        features_df['Special_char_ratio_in_URL'] = features_df['url'].apply(ratio_special_chars)
        features_df['English_words_in_URL'] = features_df['url'].apply(english_words_count)
        features_df['Avg_english_word_len_in_URL'] = features_df['url'].apply(avg_english_word_length)
        features_df['Avg_word_len_in_URL'] = features_df['url'].apply(avg_word_length)
        features_df['Is_shortened_URL'] = features_df['url'].apply(is_shortened_url)
        features_df['Path_len'] = features_df['url'].apply(path_length)
        features_df['Query_len'] = features_df['url'].apply(query_length)
        features_df['Num_params'] = features_df['url'].apply(count_params)
        features_df['Num_unique_chars'] = features_df['url'].apply(count_unique_chars)
        features_df['Max_consecutive_chars'] = features_df['url'].apply(max_consecutive_chars)
        features_df['TLD_len'] = features_df['url'].apply(tld_length)
        features_df['Subdomain_len'] = features_df['url'].apply(subdomain_length)
        features_df['Contains_suspicious_keywords'] = features_df['url'].apply(contains_suspicious_keywords)

        # Set pandas options to display all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)

        print(features_df)

if __name__ == '__main__':
    unittest.main()
