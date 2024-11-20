import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl 

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('stopwords')

class NewsDataProcessor:

    """
    Data class to fetch raw data and process it
    """

    def __init__(self):
        self.fetch_raw_data()
        self.stop_words = set(stopwords.words('english'))

    def fetch_raw_data(self):
        """
        Fetch raw data
        """
        articles = pd.read_json('../../../data/test/raw/election_news.json')
        # print(articles.head())
        self.df_data = articles

    def process_data(self):
        """
        Filter and process raw data
        Args:
            only_US: True to only filter data coming from US
            only_english: True to only filter data where tweets are in English
        
        Returns:
            df_processed: Dataframe with 2 columns: i) timestamp ii) textdata
        """
        
        df_processed = self.df_data.copy()
        df_processed = df_processed.rename(columns = {'published_at': 'timestamp'}) #rename publishedAt to timestamp
        df_processed['textdata'] = df_processed.apply(lambda row: f"{row['title']} . {row['description']}" if row['description'] != '' else row['title'], axis=1)
        
        df_processed['textdata'] = df_processed['textdata'].str.lower()
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'kamala harris', 'harris', regex=True)
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'donald trump', 'trump', regex=True)
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'kamala', '', regex=True)
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'donald', '', regex=True)
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'\bkamal\b|\bkama\b|\bkamaala\b|\bkamla\b|\bkam\b|\bkamil\b|\bkamara\b', '', regex=True)
        df_processed['textdata'] = df_processed['textdata'].str.replace(r'\bdonal\b|\donalad\b|\bdona\b', '', regex=True)



        df_processed['textdata'] = df_processed['textdata'].str.replace(' . ', ' _SEPARATOR_ ')

        # Remove stopwords from 'textdata'
        df_processed['textdata'] = df_processed['textdata'].apply(self.remove_stopwords)

        df_processed['textdata'] = df_processed['textdata'].str.replace(' _SEPARATOR_ ', ' . ')

        mask = df_processed['textdata'].str.contains('trump|harris', case=False, na=False) & \
            df_processed['description'].str.contains('trump|harris', case=False, na=False)
        
        df_processed = df_processed[mask]
        df_processed.reset_index(drop=True, inplace=True)
        
        columns_to_keep = ['textdata', 'timestamp']
        df_processed = df_processed.loc[:, columns_to_keep] #keep just textdata and timestamp columns

        json_string = df_processed.to_json(orient='records')
        json_string = f"[{json_string}]"

        with open('../../../data/test/processed/processed_election_news.json', 'w') as json_file:
            json_file.write(json_string)
        
        # print(df_processed)

        return df_processed
        
    def remove_stopwords(self, text):
        """
        Removes stopwords from the given text.
        """
        # Tokenize the text into individual words
        words = word_tokenize(text)
        
        # Filter out stopwords and non-alphabetic words (like punctuation)
        filtered_words = [word for word in words if word.lower() not in self.stop_words or word == '_SEPARATOR_']
        
        # Return the filtered text
        return " ".join(filtered_words)

processor = NewsDataProcessor()
processor.process_data()

