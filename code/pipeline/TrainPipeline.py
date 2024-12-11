import pandas as pd
import os
import sys
import re

sys.path.append('../process/train/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler, EnhancedSecondLayerDataHandler
from sentiment_analysis import SentimentAnalysis

sys.path.append('../model/')
from model import ProbabilityModeler, EnhancedProbabilityModeler

import numpy as np

LAYER2_DATAHANDLER_MAP = {'2RATIOS': SecondLayerDataHandler,
                          '1RATIO' : OneRatioSecondLayerDataHandler}

PROBABILITY_MODEL_MAP = {'OLS': EnhancedProbabilityModeler,
                         'Ridge': EnhancedProbabilityModeler,
                         'Lasso': EnhancedProbabilityModeler,
                         'Gradient Boosting': EnhancedProbabilityModeler,
                         'SARIMAX': EnhancedProbabilityModeler}

class TrainModelPipeline:


    LOCAL_INPUT_PROCESSED_DATAFILE = '../../data/train/processed/processed_data.csv'
    BOX_INPUT_PROCESSED_DATAFILE = 'https://uofi.box.com/shared/static/w3ndz27uj17hfgloq7djgyavf0jdi78v.csv?raw=1'

    def __init__(self,
                 sentiment_model_name: str,
                 layer2_process_name: str,
                 probability_model_name: str,
                 sentiment_threshold: float = 0.05,
                 run_sentiment_model: bool = True,
                 label_type: str = '538',
                 trade_type: str = 'close',
                 use_box: bool = True,
                 base_handler: bool = True,
                 **kwargs):
        
        self.sentiment_model_name = sentiment_model_name
        self.layer2_process_name = layer2_process_name
        self.probability_model_name = probability_model_name
        self.sentiment_threshold = sentiment_threshold
        self.run_sentiment_model = run_sentiment_model
        self.label_type = label_type
        self.trade_type = trade_type
        self.base_handler = base_handler
        self.kwargs = kwargs

        #Fetch the models
        self.fetch_models()
        self.apply_pipeline()

    def fetch_models(self):
        self.layer2_processor = LAYER2_DATAHANDLER_MAP.get(self.layer2_process_name, None) if self.base_handler else EnhancedSecondLayerDataHandler
        print(f"Using {self.layer2_processor}")
        self.probability_model = PROBABILITY_MODEL_MAP.get(self.probability_model_name, None)

        if self.layer2_processor is None:
            raise(Exception(f'Layer2 DataHandler:{self.layer2_process_name} is not recognized'))
        if self.probability_model is None:
            raise(Exception(f'Probability Model:{self.probability_model_name} is not recognized'))

    def apply_pipeline(self):
        #2) Apply the sentiment model
        if self.run_sentiment_model:
            output_sentiment_file = f'../../data/train/processed/{self.sentiment_model_name}_processed_data.csv'
            self.sentiment_analyzer = SentimentAnalysis(input_file_path = self.BOX_INPUT_PROCESSED_DATAFILE if self.use_box else self.LOCAL_INPUT_PROCESSED_DATAFILE,
                                                   output_file_path = output_sentiment_file,
                                                   model  = self.sentiment_model_name,
                                                   threshold = self.sentiment_threshold)
            self.sentiment_analyzer.process_tweets()

        #3/4) Apply Layer2DataHandler and apply probability model
        self.prob_modeler = self.probability_model(model_type = "Train",
                                              sentiment_model = self.sentiment_model_name,
                                              model_name = self.probability_model_name,
                                              layer2_datahandler = self.layer2_processor,
                                              **self.kwargs)
        self.prob_modeler.run_model()
        

if __name__ == "__main__":
    # # Train Model with VADER e.g. 1
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'OLS',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)

    # #Train Model with VADER e.g. 2 (Do not run sentiment model again)
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '1RATIO',
    #                                       probability_model_name = 'OLS',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)
    
    # #Train Model with VADER and Gradient Boosting
    # train_model_pipeline  = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'Gradient Boosting',
    #                                       run_sentiment_model = False,
    #                                       use_box = True,
    #                                       base_handler=True,
    #                                       n_estimators=50, max_depth=2)
    
    # # Train Model with VADER, Gradient Boosting, and Filters
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name='VADER',
    #                                           layer2_process_name='2RATIOS',
    #                                           probability_model_name='Gradient Boosting',
    #                                           run_sentiment_model=False,
    #                                           use_box=True,
    #                                           base_handler=False,
    #                                           n_estimators=50, max_depth=2)
    
    # # Train Model with VADER and Ridge
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'Ridge',
    #                                       run_sentiment_model = False,
    #                                       use_box = True,
    #                                       alpha = 0.1)
    
    # # Train Model with VADER and Lasso
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'Lasso',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)
    
    # # Train Model with VADER and SARIMAX
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'VADER',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'SARIMAX',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)
    
    #Train Model with TextBlob 
    train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'TextBlob',
                                          layer2_process_name = '2RATIOS',
                                          probability_model_name = 'OLS',
                                          run_sentiment_model = False,
                                          use_box = True)
    
    # Train Model with TextBlob and Gradient Boosting
    train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'TextBlob',
                                          layer2_process_name = '2RATIOS',
                                          probability_model_name = 'Gradient Boosting',
                                          run_sentiment_model = False,
                                          use_box = True,
                                          n_estimators=50, max_depth=2)
    
    # # Train Model with TextBlob, Gradient Boosting, and Filters
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name='TextBlob',
    #                                           layer2_process_name='2RATIOS',
    #                                           probability_model_name='Gradient Boosting',
    #                                           run_sentiment_model=False,
    #                                           use_box=True,
    #                                           base_handler=False,
    #                                           n_estimators=50, max_depth=2,
    #                                           )
    
    # # Train Model with TextBlob and Ridge
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'TextBlob',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'Ridge',
    #                                       run_sentiment_model = False,
    #                                       use_box = True,
    #                                       alpha = 0.1)
    
    # # Train Model with TextBlob and Lasso
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'TextBlob',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'Lasso',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)
    
    # # Train Model with TextBlob and SARIMAX
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name = 'TextBlob',
    #                                       layer2_process_name = '2RATIOS',
    #                                       probability_model_name = 'SARIMAX',
    #                                       run_sentiment_model = False,
    #                                       use_box = True)

    #Train Model with ABSA. Running sentiment model can take approximately 2-4 hours and requires Pytorch + CUDA.
    # So, ABSA sentiment analysis data has been pre-run and results are saved in a csv in the cloud (UIUC Box).
    # train_model_pipeline = TrainModelPipeline(sentiment_model_name='ABSA',
    #                                           layer2_process_name='2RATIOS',
    #                                           probability_model_name='OLS',
    #                                           run_sentiment_model=False,
    #                                           use_box = True)
