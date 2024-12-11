import pandas as pd
from process_label_data import LabelDataProcessor, LabelTestDataProcessor
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
                 label_type: str = 'PredictIt',
                 trade_type: str = 'close',
                 test_sentiment_model: str = 'ABSA'):
        self.sentiment_model = sentiment_model
        self.df_sentiment_data = self.get_sentiment_data()
        self.df_label_data = LabelDataProcessor.get_label_data(label_type = label_type, trade_type = trade_type)
        self.df_all_data = self.format_predictor()

    def get_sentiment_data(self):
        df_sentiment_data = pd.read_csv(f'../../data/train/processed/{self.sentiment_model}_processed_data.csv')
        df_sentiment_data.index = pd.to_datetime(df_sentiment_data['timestamp']).dt.date
        df_sentiment_data['date'] = df_sentiment_data.index
        return df_sentiment_data

    def format_predictor(self):
        positive_label = {'Neutral' : 0, 'Positive': 1, 'Negative': -1}
        df_data = self.df_sentiment_data.copy()
        df_data['istrump'] = np.where(df_data['candidate'] == 'trump', 1, 0)
        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        df_grouped_data = df_data.groupby(['date', 'candidate']).aggregate({'sentiment_indic' : ['sum', 'count']}).reset_index().set_index(['date', 'candidate'])
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        df_grouped_data['ratio'] = df_grouped_data['sum']/df_grouped_data['count']
        df_predict_data = df_grouped_data.reset_index().pivot(index = 'date', columns = 'candidate', values = 'ratio')
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index = True, right_index = True)
        return df_all_data

    def __repr__(self):
        return f'SecondLayerDataHandler:{self.DATANAME}'

class OneRatioSecondLayerDataHandler(SecondLayerDataHandler):
    DATANAME = '1RATIO'

    def format_predictor(self):
        positive_label = {'Neutral' : 0, 'Positive': 1, 'Negative': -1}
        df_data = self.df_sentiment_data.copy()
        df_data = df_data.loc[df_data['candidate'] == 'trump']
        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        df_grouped_data = df_data.groupby(['date']).aggregate({'sentiment_indic' : ['sum', 'count']}).reset_index().set_index(['date'])
        df_grouped_data.columns = df_grouped_data.columns.get_level_values(1)
        df_grouped_data['ratio'] = df_grouped_data['sum']/df_grouped_data['count']
        df_predict_data = df_grouped_data[['ratio']]
        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index = True, right_index = True)
        return df_all_data

class EnhancedSecondLayerDataHandler(SecondLayerDataHandler):
    
    def __init__(self, 
                 sentiment_model: str, 
                 label_type: str = '538',
                 trade_type: str = 'close',
                 y_var: str = 'p_trump_win',
                 min_likes: int = 5,
                 min_followers: int = 5,
                 min_account_age: int = 30,
                 english_only: bool = True):
        self.y_var = y_var
        self.min_likes = min_likes
        self.min_followers = min_followers
        self.min_account_age = min_account_age
        self.english_only = english_only
        super().__init__(sentiment_model,
                         label_type = label_type,
                         trade_type = trade_type)
    
    def format_predictor(self):
        positive_label = {'Neutral' : 0, 'Positive': 1, 'Negative': -1}
        df_data = self.df_sentiment_data.copy()

        df_data['sentiment_indic'] = df_data['sentiment_label'].map(positive_label)
        
        # account age
        user_join = pd.to_datetime(df_data['user_join_date'])
        timestamp = pd.to_datetime(df_data['timestamp'])
        df_data['account_age'] = (timestamp - user_join).dt.days

        # filtering
        df_data = df_data[df_data['account_age'] > self.min_account_age]
        df_data = df_data[df_data['user_followers_count'] > self.min_followers]
        df_data = df_data[df_data['likes'] > self.min_likes]

        if self.english_only:
            df_data = df_data[df_data['is_en']]

        # retain sentiment score, likes, account age, user followers count
        df_grouped_data = df_data.groupby(['date', 'candidate']).aggregate({
            'sentiment_indic': 'mean',
        }).reset_index().set_index(['date', 'candidate'])

        df_grouped_data['rolling_sentiment_indic'] = df_grouped_data['sentiment_indic'].rolling(window=7, min_periods=1).mean()
        df_grouped_data['ratio'] = df_grouped_data['sentiment_indic'] / (df_grouped_data['sentiment_indic'].abs() + 1)

        pivot_fields = [
            'ratio',
            'sentiment_indic',
            'rolling_sentiment_indic',
        ]
        pivoted_dfs = {}
        for field in pivot_fields:
            pivoted_dfs[field] = df_grouped_data.reset_index().pivot(index='date', columns='candidate', values=field)
        
        df_predict_data = pivoted_dfs['ratio']
        for field, pivoted_df in pivoted_dfs.items():
            if field != 'ratio':
                df_predict_data = df_predict_data.join(pivoted_df, rsuffix=f'_{field}')

        df_all_data = pd.merge(df_predict_data, self.df_label_data[['p_trump_win']], left_index=True, right_index=True, how='inner')
        df_all_data[f'p_trump_win_rolling'] = df_all_data['p_trump_win'].rolling(window=7, min_periods=1).mean()

        return df_all_data
