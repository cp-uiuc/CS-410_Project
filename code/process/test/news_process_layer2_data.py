import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from news_process_label_data import NewsLabelDataProcessor

class NewsSecondLayerDataHandler:
    DATANAME = '2RATIOS'

    def __init__(self,
                 sentiment_model: str,
                 label_type: str = '538',
                 trade_type: str = 'close'):
        self.sentiment_model = sentiment_model
        self.label_type = label_type
        self.trade_type = trade_type
        self.df_sentiment_data = self.get_sentiment_data()
        self.df_label_data = NewsLabelDataProcessor.get_label_data(label_type = label_type, trade_type = trade_type)
        self.df_all_data = self.format_predictor()

    def get_sentiment_data(self):
        # Read the test data (news articles) from the CSV file
        df_sentiment_data = pd.read_csv(f'../../data/test/processed/{self.sentiment_model}_processed_news_data.csv')
        df_sentiment_data['timestamp'] = pd.to_datetime(df_sentiment_data['timestamp'], unit='ms')
        df_sentiment_data['date'] = df_sentiment_data['timestamp'].dt.date
        return df_sentiment_data

    def format_predictor(self):
        # Label mapping for sentiment scores (positive, negative, neutral)
        positive_label = {'neutral': 0, 'positive': 1, 'negative': -1}
        
        # Make a copy of the sentiment data
        df_data = self.df_sentiment_data.copy()

        # Map sentiment labels to numeric indicators for Trump and Harris
        df_data['trump_sentiment_indic'] = df_data['trump_sentiment_label'].map(positive_label)
        df_data['harris_sentiment_indic'] = df_data['harris_sentiment_label'].map(positive_label)

        # Calculate sentiment sums and counts grouped by date and candidate mentions
        df_grouped_data = df_data.groupby('date').aggregate({
            'trump_sentiment_indic': ['sum', 'count'],
            'harris_sentiment_indic': ['sum', 'count']
        }).reset_index().set_index('date')

        # Flatten multi-level columns
        df_grouped_data.columns = ['_'.join(col).strip() for col in df_grouped_data.columns]

        # Calculate sentiment ratios
        df_grouped_data['trump_ratio'] = df_grouped_data['trump_sentiment_indic_sum'] / df_grouped_data['trump_sentiment_indic_count']
        df_grouped_data['harris_ratio'] = df_grouped_data['harris_sentiment_indic_sum'] / df_grouped_data['harris_sentiment_indic_count']

        # Create the pivot-like structure to match original logic
        df_predict_data = df_grouped_data[['trump_ratio', 'harris_ratio']]

        # Merge with label data (p_trump_win)
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index=True, right_index=True)

        return df_all_data

    def __repr__(self):
        return f'NewsSecondLayerDataHandler:{self.DATANAME}'
    
class NewsOneRatioSecondLayerDataHandler(NewsSecondLayerDataHandler):
    DATANAME = '1RATIO'

    def format_predictor(self):
        positive_label = {'neutral': 0, 'positive': 1, 'negative': -1}

        # Make a copy of the data
        df_data = self.df_sentiment_data.copy()

        # Filter for Trump-related data
        df_data = df_data.loc[df_data['mentions_trump'] == 1]

        # Map sentiment labels to numerical indicators for Trump
        df_data['trump_sentiment_indic'] = df_data['trump_sentiment_label'].map(positive_label)

        # Group by timestamp (date part only) and calculate sum and count of sentiment indicators
        df_data['date'] = pd.to_datetime(df_data['timestamp']).dt.date
        df_grouped_data = df_data.groupby('date').aggregate({
            'trump_sentiment_indic': ['sum', 'count']
        }).reset_index().set_index('date')

        # Flatten MultiIndex columns
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        
        # Calculate sentiment ratio
        df_grouped_data['ratio'] = df_grouped_data['sum'] / df_grouped_data['count']
        
        # Filter only the ratio for prediction input
        df_predict_data = df_grouped_data[['ratio']]
        
        # Merge with label data (assumes self.df_label_data has 'p_trump_win' indexed by date)
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], 
                            left_index=True, right_index=True)
        
        return df_all_data
        
# if __name__ == "__main__":
#     sentiment_model = "ABSA"  # Assuming you're using ABSA for sentiment analysis
#     test_handler = NewsOneRatioSecondLayerDataHandler(sentiment_model,
#                                        label_type = '538',
#                                        trade_type = 'close')

#     #Inserting option for PredictIt data
#     predictit_test_handler = NewsOneRatioSecondLayerDataHandler (sentiment_model,
#                                        label_type = 'PredictIt',
#                                        trade_type = 'close')

    
#     # This will give you the processed test data
#     df_test_data = test_handler.df_all_data  # DataFrame with 'date' and 'ratio' columns
    
#     # Save the test data to a CSV file
#     df_test_data.to_csv('../../../data/test/processed/test_news_features_2.csv', index=True)