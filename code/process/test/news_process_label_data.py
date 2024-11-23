import pandas as pd
import numpy as np

class NewsLabelDataProcessor:

    """
    Load the test label data. 
    label_type: 
        538: Data found in daily_summary.csv file from https://projects.fivethirtyeight.com/2024-election-forecast/
        PredictIt: Data found on predictit
    """

    def __init__(self,
                 label_type: str = '538',
                 trade_type:str = 'close'):
        
        self.label_type = label_type
        self.trade_type = trade_type
        self.df_label_data = self.get_label_data(label_type = self.label_type,
                                                 trade_type = self.trade_type)

    @staticmethod
    def get_label_data(self):
        """
        Fetch raw data
        """
        label_data = pd.read_csv('../../../data/test/raw/daily_summary.csv')
        label_data = label_data[(label_data['state_abb'].isna()) & (label_data['variable'] == 'electoral college') & (label_data['metric'] == 'p_win')]
        label_data = label_data.pivot(index='model_date', columns='party', values='value')
        label_data = label_data.rename(columns = {'REP': 'p_trump_win', 'DEM': 'p_trump_lose_1', 'IND': 'p_trump_lose_2'})
        label_data.fillna(0, inplace=True)
        label_data['p_trump_lose'] = label_data['p_trump_lose_1'] + label_data['p_trump_lose_2']
        return label_data[['p_trump_win', 'p_trump_lose']]

    @staticmethod
    def get_label_data(label_type:str, trade_type: str):
        if label_type == '538':
            df_label = NewsLabelDataProcessor.get_538_label_data()

        if label_type == 'PredictIt':
            df_label = NewsLabelDataProcessor.get_PredictIt_data(trade_type = trade_type)
        return df_label

    @staticmethod
    def get_538_label_data():
        label_data = pd.read_csv('../../../data/test/raw/daily_summary.csv')
        label_data = label_data[(label_data['state_abb'].isna()) & (label_data['variable'] == 'electoral college') & (label_data['metric'] == 'p_win')]
        label_data = label_data.pivot(index='model_date', columns='party', values='value')
        label_data = label_data.rename(columns = {'REP': 'p_trump_win', 'DEM': 'p_trump_lose_1', 'IND': 'p_trump_lose_2'})
        label_data.fillna(0, inplace=True)
        label_data['p_trump_lose'] = label_data['p_trump_lose_1'] + label_data['p_trump_lose_2']
        label_data.index = pd.to_datetime(label_data.index)
        return label_data[['p_trump_win', 'p_trump_lose']]

    @staticmethod
    def get_PredictIt_data(trade_type: str = 'close'):
        label_data = pd.read_csv('../../../data/test/raw/2024_predictit_data.csv')
        label_data = label_data.rename(columns = {'Date (ET)': 'modeldate'})
        label_data.index = pd.to_datetime(label_data['modeldate'])
        if trade_type == 'close':
            label_data = label_data.rename(columns = {'Close Share Price': 'p_trump_win'})
            label_data['p_trump_lose'] = 1 - label_data['p_trump_win']

        elif trade_type == 'average':
            label_data = label_data.rename(columns = {'Average Trade Price': 'p_trump_win'})
            label_data['p_trump_lose'] = 1 - label_data['p_trump_win']

        else:
            raise(Exception(f'Unrecognized trade type: {trade_type}'))
            
        return label_data[['p_trump_win', 'p_trump_lose']]
