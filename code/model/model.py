import pandas as pd
import os
import sys
import re

sys.path.append('../process/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler, EnhancedSecondLayerDataHandler
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score

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
                 label_type:str = '538',
                 trade_type: str = 'close',
                 verbose: bool = True):
        self.sentiment_model = sentiment_model
        self.label_type = label_type
        self.trade_type = trade_type
        self.layer2_datahandler = layer2_datahandler(sentiment_model = self.sentiment_model,
                                                     label_type = self.label_type,
                                                     trade_type = self.trade_type)
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
                 verbose: bool = True, 
                 **kwargs):
        self.layer2_datahandler = layer2_datahandler
        self.verbose = verbose
        self.model_type = model_type
        self.kwargs = kwargs

    def run_model(self, y_var: str = 'p_trump_win', rolling = False):
        original_y_var = y_var
        y_var = f"{y_var}_rolling" if rolling else y_var
        df_data = self.layer2_datahandler.df_all_data
        x_vars = [col for col in df_data.columns if col != y_var and col != original_y_var]
        X = df_data[x_vars]
        y = df_data[y_var]

        df_test_data = self.layer2_datahandler.df_test_data
        X_test = df_test_data[x_vars]
        y_test = df_test_data[y_var]

        if self.model_type == 'OLS' or self.model_type == 'Linear Regression':
            model = LinearRegression().fit(X, y)
        elif self.model_type == 'Ridge':
            model = Ridge(**self.kwargs).fit(X, y)
        elif self.model_type == 'Lasso':
            model = Lasso(**self.kwargs).fit(X, y)
        elif self.model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(**self.kwargs).fit(X, y)
        elif self.model_type == 'SARIMAX':
            exog_vars = df_data[['other', 'trump', 'other_sentiment_indic', 'trump_sentiment_indic']]
            model = SARIMAX(y, exog = exog_vars, order = (1, 1, 1), seasonal_order = (0, 0, 0, 0)).fit()
            # print(model.summary())
            # y_pred = model.predict(start=len(y), end=len(y) + len(y_test) - 1, exog = X_test[['other', 'trump', 'other_sentiment_indic', 'trump_sentiment_indic']])
            # predictions = pd.DataFrame({'y_test': y_test.values, 'y_pred': y_pred.values})
            # print(predictions.head())
            # print(f"Modelname: {self.model_type}, Sentiment Model: {self.layer2_datahandler.sentiment_model}, rmse: {mean_squared_error(y_test.values, y_pred.values)}")
        else:
            raise ValueError(f'Invalid model_type: {self.model_type}')
        
        if self.model_type == 'SARIMAX':
            prediction = model.predict(start=len(y), end=len(y) + len(y_test) - 1, exog = X_test[['other', 'trump', 'other_sentiment_indic', 'trump_sentiment_indic']])
        else:
            prediction = model.predict(X_test)

        # if y value > 0.5, predict 1, else predict 0
        binary_prediction = np.where(prediction > 0.5, 1, 0)
        binary_y_test = np.where(y_test > 0.5, 1, 0)

        # combine y_test and prediction into a dataframe
        df = pd.DataFrame({'y_test': binary_y_test, 'prediction': binary_prediction})
        print(df.head())
        print(df.tail())
        # print count of 0 and 1 in y_test and prediction
        print(df['y_test'].value_counts())
        print(df['prediction'].value_counts())

        mae, rmse, r2, mape = self.evaluate_model(binary_y_test, binary_prediction)

        if self.verbose:
            print(f'Modelname: {self.model_type}, Sentiment Model: {self.layer2_datahandler.sentiment_model}, kwargs: {self.kwargs}, r2: {r2}, mape: {mape}, mae: {mae}, rmse: {rmse}')
        return model
    
    def evaluate_model(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, rmse, r2, mape



if __name__ == "__main__":

    # prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = SecondLayerDataHandler)
    # one_var_prob_modeler = ProbabilityModeler(sentiment_model= 'VADER', layer2_datahandler = OneRatioSecondLayerDataHandler)

    # model_pipeline = [prob_modeler, one_var_prob_modeler]

    # for modeler in model_pipeline:
    #     model = modeler.run_model()

    # enhanced modeling
    #1) Using 538 predictions
    # layer2_datahandler = EnhancedSecondLayerDataHandler(sentiment_model='VADER',
    #                                                     label_type = '538',
    #                                                     trade_type = 'close',
    #                                                     min_likes = 10,
    #                                                     min_followers = 10,
    #                                                     min_account_age = 10,
    #                                                     english_only = True)
    #2) Using PredictIt
    layer2_datahandler = EnhancedSecondLayerDataHandler(sentiment_model='VADER',
                                                        label_type = 'PredictIt',
                                                        trade_type = 'close',
                                                        min_likes = 0,
                                                        min_followers = 0,
                                                        min_account_age = 0,
                                                        english_only = False)
    prob_modeler_ols = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='OLS')
    prob_modeler_sarimax = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='SARIMAX')
    prob_modeler_ridge1 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Ridge', alpha=0.1)
    prob_modeler_ridge3 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Ridge', alpha=0.3)
    prob_modeler_ridge6 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Ridge', alpha=0.6)
    prob_modeler_ridge8 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Ridge', alpha=0.8)
    # prob_modeler_lasso01 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Lasso', alpha=0.1)
    prob_modeler_gb1 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Gradient Boosting', n_estimators=50, max_depth=2)
    prob_modeler_gb2 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Gradient Boosting', n_estimators=20, max_depth=2)
    prob_modeler_gb3 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Gradient Boosting', n_estimators=100, max_depth=2)
    
    model_pipeline = [
        prob_modeler_sarimax,
        prob_modeler_ols, 
        prob_modeler_ridge1,
        prob_modeler_ridge3,
        prob_modeler_ridge6,
        prob_modeler_ridge8, 
        # prob_modeler_lasso01, 
        prob_modeler_gb1,
        prob_modeler_gb2,
        prob_modeler_gb3
    ]

    for modeler in model_pipeline:
        print("--------------------------------RUNNING MODEL--------------------------------")
        model = modeler.run_model()