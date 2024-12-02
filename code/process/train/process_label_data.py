import pandas as pd
import os
import re

class LabelDataProcessor:

    """
    Load the train label data

    Args:
        label_type: Label data type. Can be 538 or PredictIt data
    """

    def __init__(self,
                 label_type: str = '538',
                 trade_type:str = 'close'):
        self.label_type = label_type
        self.trade_type = trade_type
        self.df_label_data = self.get_label_data(label_type = self.label_type,
                                                 trade_type = self.trade_type)

    @staticmethod
    def get_label_data(label_type:str, trade_type: str):
        if label_type == '538':
            df_label = LabelDataProcessor.get_538_label_data()

        if label_type == 'PredictIt':
            df_label = LabelDataProcessor.get_PredictIt_data(trade_type = trade_type)
        return df_label

    @staticmethod
    def get_538_label_data():
        label_data = pd.read_csv('../../data/train/raw/presidential_national_toplines_2020.csv')
        label_data.index = pd.to_datetime(label_data['modeldate'])
        label_data = label_data.rename(columns = {'ecwin_inc': 'p_trump_win', 'ecwin_chal': 'p_trump_lose'})
        return label_data[['p_trump_win', 'p_trump_lose']]

    @staticmethod
    def get_PredictIt_data(trade_type: str = 'close'):
        label_data = pd.read_csv('../../data/train/raw/2020_predictit_data.csv')
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



