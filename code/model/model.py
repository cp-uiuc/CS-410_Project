import pandas as pd
import os
import sys
import re

sys.path.append('../process/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler, EnhancedSecondLayerDataHandler
import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

class ProbabilityModeler:

    """
    Probability modeler class to predict Trump probability.
    This is the baseclass used for probability modelling. 

    Subclasses can change up the run_model method to implement its own model
    """

    MODELNAME = "OLS"
    
    def __init__(self,
                 sentiment_model: str,
                 layer2_datahandler: SecondLayerDataHandler,
                 verbose: bool = True):
        self.sentiment_model = sentiment_model
        self.layer2_datahandler = layer2_datahandler(sentiment_model = self.sentiment_model)
        self.verbose = verbose

    def __repr__(self):
        return f'ProbabilityModeler:{self.MODELNAME}'

    @property
    def params_outputfile(self):
        params_dir = '../params'
        os.makedirs(params_dir, exist_ok = True)
        return f'{params_dir}/{self.sentiment_model}_{self.layer2_datahandler.DATANAME}_{self.MODELNAME}_params.csv'

    def run_model(self, y_var: str = 'p_trump_win'):
        y_var = 'p_trump_win'
        df_data = self.layer2_datahandler.df_all_data
        x_vars = [col for col in df_data.columns if col != y_var]
        model = sm.OLS(df_data[y_var], df_data[x_vars]).fit()
        if self.verbose:
            print(f'Modelname: {self.MODELNAME},Sentiment Model: {self.sentiment_model},  rsquared: {model.rsquared:.2f}')
        self.save_model(model)
        return model

    def save_model(self, model):
        params_df = pd.DataFrame(model.params).T
        params_df.index = ['betas']
        params_df.to_csv(self.params_outputfile)
        print(f'File saved in: {self.params_outputfile}')

class EnhancedProbabilityModeler(ProbabilityModeler):
    """
    Enhanced ProbabilityModeler with support for Ridge, Lasso, and Gradient Boosting.
    """

    def __init__(self,
                 layer2_datahandler,
                 model_type: str = 'OLS',
                 alpha: float = 1.0,
                 verbose: bool = True):
        self.layer2_datahandler = layer2_datahandler
        self.verbose = verbose
        self.model_type = model_type
        self.alpha = alpha

    def run_model(self, y_var: str = 'p_trump_win', rolling = True):
        y_var = f"{y_var}_rolling" if rolling else y_var
        df_data = self.layer2_datahandler.df_all_data
        x_vars = [col for col in df_data.columns if col != y_var]
        X = df_data[x_vars]
        y = df_data[y_var]

        if self.model_type == 'OLS':
            model = sm.OLS(y, X).fit()
        elif self.model_type == 'Ridge':
            model = Ridge(alpha=self.alpha).fit(X, y)
        elif self.model_type == 'Lasso':
            model = Lasso(alpha=self.alpha).fit(X, y)
        elif self.model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor().fit(X, y)
        else:
            raise ValueError(f'Invalid model_type: {self.model_type}')

        if self.verbose:
            print(f'Modelname: {self.model_type},Sentiment Model: {self.layer2_datahandler.sentiment_model},  rsquared: {model.score(X, y) if self.model_type != "OLS" else model.rsquared:.2f}')
        return model


if __name__ == "__main__":

    # prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = SecondLayerDataHandler)
    # one_var_prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = OneRatioSecondLayerDataHandler)

    # model_pipeline = [prob_modeler, one_var_prob_modeler]

    # for modeler in model_pipeline:
    #     model = modeler.run_model()

    # enhanced modeling
    layer2_datahandler = EnhancedSecondLayerDataHandler(sentiment_model='VADER')
    prob_modeler_ols = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='OLS')
    prob_modeler_ridge = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Ridge', alpha=0.1)
    prob_modeler_lasso = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Lasso', alpha=10.0)
    prob_modeler_gb = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Gradient Boosting')

    # prob_modeler_ols.plot_win_and_sentiment()
    # prob_modeler_ols.plot_win_percentage()
    # prob_modeler_ols.plot_sentiment()
    
    model_pipeline = [
        prob_modeler_ols, 
        prob_modeler_ridge, 
        prob_modeler_lasso, 
        prob_modeler_gb
    ]

    for modeler in model_pipeline:
        print("--------------------------------RUNNING MODEL--------------------------------")
        model = modeler.run_model()