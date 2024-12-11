import pandas as pd
import os
import sys
import re

sys.path.append('../process/test/')
from news_process_layer2_data import NewsSecondLayerDataHandler, NewsOneRatioSecondLayerDataHandler
from news_sentiment_analysis import NewsSentimentAnalysis

sys.path.append('../model/')
from model import ProbabilityModeler

import numpy as np

LAYER2_DATAHANDLER_MAP = {'2RATIOS': NewsSecondLayerDataHandler,
                          '1RATIO' : NewsOneRatioSecondLayerDataHandler}

PROBABILITY_MODEL_MAP = {'OLS': ProbabilityModeler}

class TestModelPipeline:


    INPUT_PROCESSED_DATAFILE = '../../data/test/processed/processed_election_news.json'

    def __init__(self,
                 sentiment_model_name: str,
                 layer2_process_name: str,
                 probability_model_name: str,
                 sentiment_threshold: float = 0.05,
                 run_sentiment_model: bool = True,
                 label_type: str = '538',
                 trade_type: str = 'close'):
        
        self.sentiment_model_name = sentiment_model_name
        self.layer2_process_name = layer2_process_name
        self.probability_model_name = probability_model_name
        self.sentiment_threshold = sentiment_threshold
        self.run_sentiment_model = run_sentiment_model
        self.label_type = label_type
        self.trade_type = trade_type

        #Fetch the models
        self.fetch_models()
        self.apply_pipeline()

    def fetch_models(self):
        self.layer2_processor = LAYER2_DATAHANDLER_MAP.get(self.layer2_process_name, None)
        self.probability_model = PROBABILITY_MODEL_MAP.get(self.probability_model_name, None)

        if self.layer2_processor is None:
            raise(Exception(f'Layer2 DataHandler:{self.layer2_process_name} is not recognized'))
        if self.probability_model is None:
            raise(Exception(f'Probability Model:{self.probability_model_name} is not recognized'))

    def apply_pipeline(self):
        #2) Apply the sentiment model
        if self.run_sentiment_model:
            output_sentiment_file = f'../../data/test/processed/{self.sentiment_model_name}_processed_data.csv'
            self.sentiment_analyzer = NewsSentimentAnalysis(input_file_path = self.INPUT_PROCESSED_DATAFILE,
                                                   output_file_path = output_sentiment_file,
                                                   model  = self.sentiment_model_name,
                                                   threshold = self.sentiment_threshold)
            self.sentiment_analyzer.process_news()

        #3/4) Apply Layer2DataHandler and apply probability model
        self.prob_modeler = self.probability_model(sentiment_model= self.sentiment_model_name,
                                              layer2_datahandler = self.layer2_processor, model_type='Test')
        self.prob_modeler.run_model()
        

if __name__ == "__main__":
    #Test Model with VADER e.g. 1
    test_model_pipeline = TestModelPipeline(sentiment_model_name = 'VADER',
                                          layer2_process_name = '2RATIOS',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = True)

    #Test Model with VADER e.g. 2 (Do not run sentiment model again)
    test_model_pipeline  = TestModelPipeline(sentiment_model_name = 'VADER',
                                          layer2_process_name = '1RATIO',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = False)
    
    #Test Model with TextBlob 
    test_model_pipeline  = TestModelPipeline(sentiment_model_name = 'TextBlob',
                                          layer2_process_name = '2RATIOS',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = True)

    #Test Model with ABSA (running sentiment model can take approximately 2-4 hours)
    test_model_pipeline  = TestModelPipeline(sentiment_model_name='ABSA',
                                              layer2_process_name='2RATIOS',
                                              probability_model_name='OLS',
                                              run_sentiment_model=False)
