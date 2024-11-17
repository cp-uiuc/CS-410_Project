import pandas as pd
import numpy as np


class NewsDataProcessor:

    """
    Data class to fetch raw data and process it
    """

    def __init__(self):
        self.fetch_raw_data()

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
        df_processed['textdata'] = df_processed.apply(lambda row: f"{row['title']} - {row['description']}" if row['description'] != '' else row['title'], axis=1)
        
        columns_to_keep = ['textdata', 'timestamp']
        df_processed = df_processed.loc[:, columns_to_keep] #keep just textdata and timestamp columns
        
        mask = df_processed['textdata'].str.contains('trump|harris', case=False, na=False)

        df_processed = df_processed[mask]
        df_processed.reset_index(drop=True, inplace=True)

        json_string = df_processed.to_json(orient='records')
        json_string = f"[{json_string}]"

        with open('../../../data/test/processed/processed_election_news.json', 'w') as json_file:
            json_file.write(json_string)
        
        # print(df_processed)

        return df_processed
        

processor = NewsDataProcessor()
processor.process_data()

