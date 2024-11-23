import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from news_process_label_data import NewsLabelDataProcessor

class NewsTestDataHandler:
    DATANAME = 'TEST_NEWS'

    def __init__(self, 
                 sentiment_model: str,
                 label_type: str = '538',
                 trade_type: str = 'close'):
        self.label_type = label_type
        self.trade_type = trade_type
        self.sentiment_model = sentiment_model
        self.df_sentiment_data = self.get_sentiment_data()
        self.df_label_data = NewsLabelDataProcessor.get_label_data(label_type = label_type, trade_type = trade_type)
        self.df_all_data = self.format_predictor()

    def get_sentiment_data(self):
        # Read the test data (news articles) from the CSV file
        df_sentiment_data = pd.read_csv(f'../../../data/test/processed/{self.sentiment_model}_processed_news_data.csv')
        df_sentiment_data['timestamp'] = pd.to_datetime(df_sentiment_data['timestamp'], unit='ms')
        df_sentiment_data['date'] = df_sentiment_data['timestamp'].dt.date
        return df_sentiment_data

    def format_predictor(self):
        # Label mapping for sentiment scores (positive, negative, neutral)
        positive_label = {'Neutral': 0, 'Positive': 1, 'Negative': -1}
        
        # Make a copy of the sentiment data
        df_data = self.df_sentiment_data.copy()
        
        # Perform sentiment analysis using VADER for each article's text
        sid = SentimentIntensityAnalyzer()
        df_data['sentiment_score'] = df_data['textdata'].apply(lambda text: sid.polarity_scores(text)['compound'])
        df_data['sentiment_label'] = df_data['sentiment_score'].apply(self.label_sentiment)

        # Map the sentiment labels to numeric values (0, 1, -1)
        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        
        # Group by 'date' and aggregate sentiment scores
        df_grouped_data = df_data.groupby(['date']).aggregate({'sentiment_indic': ['sum', 'count']}).reset_index().set_index(['date'])
        
        # Flatten multi-level columns
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        
        # Calculate ratio: sum of sentiment_indic / count of articles for each date
        df_grouped_data['ratio'] = df_grouped_data['sum'] / df_grouped_data['count']
        
        # Return just the ratio for the test set
        df_predict_data = df_grouped_data[['ratio']]

        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index = True, right_index = True)

        return df_all_data

    def label_sentiment(self, score):
        """Label sentiment based on VADER score."""
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
        
if __name__ == "__main__":
    sentiment_model = "ABSA"  # Assuming you're using ABSA for sentiment analysis
    test_handler = NewsTestDataHandler(sentiment_model,
                                       label_type = '538',
                                       trade_type = 'close')

    #Inserting option for PredictIt data
    predictit_test_handler = NewsTestDataHandler(sentiment_model,
                                       label_type = 'PredictIt',
                                       trade_type = 'close')

    
    # This will give you the processed test data
    df_test_data = test_handler.df_all_data  # DataFrame with 'date' and 'ratio' columns
    
    # Save the test data to a CSV file
    df_test_data.to_csv('../../../data/test/processed/test_news_features_2.csv', index=True)