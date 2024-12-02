import pandas as pd
from process_label_data import LabelDataProcessor
import numpy as np

class SecondLayerDataHandler:

    """
    Base second layer datahandler to format processed data
    Also:
    i) Chooses which other features to add
    ii) Chooses which tweets/newsheadlines to filter out

    This class is the base class to be used by subclasses for other data manipulation
    """

    DATANAME = '2RATIOS'
    def __init__(self, 
                 sentiment_model: str,
                 label_type: str = '538',
                 trade_type: str = 'close'):
        self.sentiment_model = sentiment_model
        self.df_sentiment_data = self.get_sentiment_data()
        self.df_label_data = LabelDataProcessor.get_label_data(label_type = label_type, trade_type = trade_type)
        self.df_all_data = self.format_predictor()

    def get_sentiment_data(self):
        df_sentiment_data = pd.read_csv(f'../../data/train/processed/{self.sentiment_model}_processed_data.csv')
        df_sentiment_data.index = pd.to_datetime(df_sentiment_data['timestamp'])
        df_sentiment_data['date'] = df_sentiment_data.index.date
        return df_sentiment_data

    def format_predictor(self):
        positive_label = {'neutral': 0, 'positive': 1, 'negative': -1}
        df_data = self.df_sentiment_data.copy()

        # Convert timestamp to date
        df_data['date'] = pd.to_datetime(df_data['timestamp']).dt.date

        # Map sentiment labels to numeric indicators for Trump and Biden
        df_data['trump_sentiment_indic'] = df_data['trump_sentiment_label'].map(positive_label)
        df_data['biden_sentiment_indic'] = df_data['biden_sentiment_label'].map(positive_label)

        # Calculate sentiment sums and counts grouped by date and candidate mentions
        df_grouped_data = df_data.groupby('date').aggregate({
            'trump_sentiment_indic': ['sum', 'count'],
            'biden_sentiment_indic': ['sum', 'count']
        }).reset_index().set_index('date')

        # Flatten multi-level columns
        df_grouped_data.columns = ['_'.join(col).strip() for col in df_grouped_data.columns]

        # Calculate sentiment ratios
        df_grouped_data['trump_ratio'] = df_grouped_data['trump_sentiment_indic_sum'] / df_grouped_data['trump_sentiment_indic_count']
        df_grouped_data['biden_ratio'] = df_grouped_data['biden_sentiment_indic_sum'] / df_grouped_data['biden_sentiment_indic_count']

        # Create the pivot-like structure to match original logic
        df_predict_data = df_grouped_data[['trump_ratio', 'biden_ratio']]

        # Merge with label data (p_trump_win)
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index=True, right_index=True)

        return df_all_data
    
    def __repr__(self):
        return f'SecondLayerDataHandler:{self.DATANAME}'

class OneRatioSecondLayerDataHandler(SecondLayerDataHandler):
    DATANAME = '1RATIO'

    def format_predictor(self):
        positive_label = {'neutral': 0, 'positive': 1, 'negative': -1}

        # Make a copy of the data
        df_data = self.df_sentiment_data.copy()

        # Filter for Trump-related data
        df_data = df_data.loc[df_data['is_en'] & (df_data['mentions_trump'] > 0)]

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


class EnhancedSecondLayerDataHandler(SecondLayerDataHandler):
    
    def __init__(self, 
                 sentiment_model: str, 
                 label_type: str = '538',
                 trade_type: str = 'close',
                 y_var: str = 'p_trump_win'):
        super().__init__(sentiment_model,
                         label_type = label_type,
                         trade_type = trade_type)
        self.y_var = y_var
    
    def format_predictor(self):
        positive_label = {'neutral' : 0, 'positive': 1, 'negative': -1}
        df_data = self.df_sentiment_data.copy()

        # Map sentiment labels to numeric indicators for both Trump and Biden
        df_data['trump_sentiment_indic'] = df_data['trump_sentiment_label'].map(positive_label)
        df_data['biden_sentiment_indic'] = df_data['biden_sentiment_label'].map(positive_label)

        # Additional feature engineering
        # Convert timestamp to date
        df_data['date'] = pd.to_datetime(df_data['timestamp']).dt.date

        # Account age calculation
        user_join = pd.to_datetime(df_data['user_join_date'])
        timestamp = pd.to_datetime(df_data['timestamp'])
        df_data['account_age'] = (timestamp - user_join).dt.days

        # Create interaction features
        df_data['trump_sentiment_followers'] = df_data['trump_sentiment_indic'] * df_data['user_followers_count']
        df_data['biden_sentiment_followers'] = df_data['biden_sentiment_indic'] * df_data['user_followers_count']
        df_data['trump_sentiment_likes'] = df_data['trump_sentiment_indic'] * df_data['likes']
        df_data['biden_sentiment_likes'] = df_data['biden_sentiment_indic'] * df_data['likes']
        df_data['trump_sentiment_account_age'] = df_data['trump_sentiment_indic'] * df_data['account_age']
        df_data['biden_sentiment_account_age'] = df_data['biden_sentiment_indic'] * df_data['account_age']
        df_data['likes_account_age'] = df_data['likes'] * df_data['account_age']
        df_data['followers_account_age'] = df_data['user_followers_count'] * df_data['account_age']

        # Sentiment volatility
        df_data['trump_sentiment_volatility'] = df_data['trump_sentiment_indic'].rolling(window=7, min_periods=1).std()
        df_data['biden_sentiment_volatility'] = df_data['biden_sentiment_indic'].rolling(window=7, min_periods=1).std()

        # Group by date and candidate
        df_grouped_data = df_data.groupby(['date', 'candidate']).aggregate({
            'trump_sentiment_indic': 'mean',
            'biden_sentiment_indic': 'mean',
            'likes': 'mean',
            'user_followers_count' : 'mean',
            'account_age': 'mean',
            'trump_sentiment_followers': 'mean',
            'biden_sentiment_followers': 'mean',
            'trump_sentiment_likes': 'mean',
            'biden_sentiment_likes': 'mean',
            'trump_sentiment_account_age': 'mean',
            'biden_sentiment_account_age': 'mean',
            'likes_account_age': 'mean',
            'followers_account_age': 'mean',
            'trump_sentiment_volatility': 'mean',
            'biden_sentiment_volatility': 'mean',
        }).reset_index().set_index(['date', 'candidate'])

        # Rolling sentiment scores for Trump and Biden
        df_grouped_data['rolling_trump_sentiment_indic'] = df_grouped_data['trump_sentiment_indic'].rolling(window=7, min_periods=1).mean()
        df_grouped_data['rolling_biden_sentiment_indic'] = df_grouped_data['biden_sentiment_indic'].rolling(window=7, min_periods=1).mean()

        # Calculate sentiment ratios
        df_grouped_data['trump_ratio'] = df_grouped_data['trump_sentiment_indic'] / (df_grouped_data['trump_sentiment_indic'].abs() + 1)
        df_grouped_data['biden_ratio'] = df_grouped_data['biden_sentiment_indic'] / (df_grouped_data['biden_sentiment_indic'].abs() + 1)

        # Pivot to create a structure for prediction
        pivot_fields = [
            'trump_ratio', 'biden_ratio', 'trump_sentiment_indic', 'biden_sentiment_indic', 'likes', 'account_age', 
            'user_followers_count', 'rolling_trump_sentiment_indic', 'rolling_biden_sentiment_indic', 
            'trump_sentiment_followers', 'biden_sentiment_followers', 'trump_sentiment_likes', 'biden_sentiment_likes', 
            'trump_sentiment_account_age', 'biden_sentiment_account_age', 'likes_account_age', 'followers_account_age', 
            'trump_sentiment_volatility', 'biden_sentiment_volatility'
        ]

        pivoted_dfs = {}
        for field in pivot_fields:
            pivoted_dfs[field] = df_grouped_data.reset_index().pivot(index='date', columns='candidate', values=field)

        # Create the final dataframe with prediction fields
        df_predict_data = pivoted_dfs['trump_ratio']
        for field, pivoted_df in pivoted_dfs.items():
            if field != 'trump_ratio':
                df_predict_data = df_predict_data.join(pivoted_df, rsuffix=f'_{field}')

        # Merge with label data (assumes self.df_label_data has 'p_trump_win' indexed by date)
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index=True, right_index=True, how='inner')

        # Rolling average for p_trump_win
        df_all_data[f'p_trump_win_rolling'] = df_all_data['p_trump_win'].rolling(window=7, min_periods=1).mean()
        
        # Day of week features
        df_all_data['day_of_week'] = df_all_data.index.dayofweek
        df_all_data['is_weekend'] = df_all_data['day_of_week'].apply(lambda x: 1 if x > 4 else 0)

        return df_all_data
