import pandas as pd
import os
import sys
import re

sys.path.append('../process/train/')
from process_layer2_data import SecondLayerDataHandler, OneRatioSecondLayerDataHandler, EnhancedSecondLayerDataHandler
sys.path.append('../process/test/')
from news_process_layer2_data import NewsSecondLayerDataHandler

import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_score

class ProbabilityModeler:

    """
    Probability modeler class to predict Trump probability.
    This is the baseclass used for probability modelling. 

    Subclasses can change up the run_model method to implement its own model
    """
    
    def __init__(self,
                 sentiment_model: str,
                 model_name: str,
                 layer2_datahandler: SecondLayerDataHandler | NewsSecondLayerDataHandler,
                 label_type:str = '538',
                 trade_type: str = 'close',
                 verbose: bool = True,
                 model_type: str = 'Train',
                 **kwargs
):
        self.sentiment_model = sentiment_model
        self.model_name = model_name
        self.label_type = label_type
        self.trade_type = trade_type
        self.model_type = model_type
        self.kwargs = kwargs
        self.layer2_datahandler = layer2_datahandler
        if self.model_type == 'Train':
            self.layer2_datahandler = SecondLayerDataHandler(sentiment_model = self.sentiment_model,
                                                        label_type = self.label_type,
                                                        trade_type = self.trade_type)
        else:
            self.layer2_datahandler = NewsSecondLayerDataHandler(sentiment_model = self.sentiment_model,
                                                        label_type = self.label_type,
                                                        trade_type = self.trade_type)
        self.verbose = verbose

    def __repr__(self):
        return f'ProbabilityModeler:{self.MODELNAME}'

    @property
    def params_outputfile(self):
        if self.model_type == 'Train':
            params_dir = '../params/train'
        else:
            params_dir = '../params/test'
        os.makedirs(params_dir, exist_ok = True)
        return f'{params_dir}/{self.sentiment_model}_{self.layer2_datahandler.DATANAME}_{self.model_name}_params.csv'

    def run_model(self, y_var: str = 'p_trump_win'):
        y_var = 'p_trump_win'
        df_data = self.layer2_datahandler.df_all_data
        df_data = df_data.fillna(0)
        x_vars = [col for col in df_data.columns if col != y_var]
        model = sm.OLS(df_data[y_var], df_data[x_vars]).fit()
        binary_y_pred = np.where(model.predict(df_data[x_vars]) > 0.5, 1, 0)
        binary_y_true = np.where(df_data[y_var] > 0.5, 1, 0)
        accuracy = accuracy_score(binary_y_true, binary_y_pred)
        if self.verbose:
            print(f'Modelname: {self.model_name},Sentiment Model: {self.sentiment_model},  accuracy: {accuracy}, rsquared: {model.rsquared:.2f}')
            # print(f"Actual Counts: {np.unique(binary_y_true, return_counts = True)}, Predicted Counts: {np.unique(binary_y_pred, return_counts = True)}")
            # # calculate accuracy for each class (other=less than 0.5, trump=greater than 0.5)
            # actual_classes = np.where(df_data[y_var] < 0.5, 1, 2)
            # predicted_classes = np.where(model.predict(df_data[x_vars]) < 0.5, 1, 2)

            # # Masks for each class in y_actual
            # class1_mask = (actual_classes == 1)
            # class2_mask = (actual_classes == 2)

            # # Accuracy for Class 1
            # class1_correct = np.sum(predicted_classes[class1_mask] == actual_classes[class1_mask])
            # class1_total = np.sum(class1_mask)
            # class1_accuracy = class1_correct / class1_total if class1_total > 0 else 0

            # # Accuracy for Class 2
            # class2_correct = np.sum(predicted_classes[class2_mask] == actual_classes[class2_mask])
            # class2_total = np.sum(class2_mask)
            # class2_accuracy = class2_correct / class2_total if class2_total > 0 else 0

            # # Print results
            # print(f"Other Accuracy: {class1_accuracy:.2f}")
            # print(f"Trump Accuracy: {class2_accuracy:.2f}")
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

    def run_model(self, y_var: str = 'p_trump_win', rolling = False):
        original_y_var = y_var
        y_var = f"{y_var}_rolling" if rolling else y_var
        df_data = self.layer2_datahandler.df_all_data
        x_vars = [col for col in df_data.columns if col != y_var and col != original_y_var]
        X = df_data[x_vars]
        y = df_data[y_var]

        # print(df_data.columns)

        # df_test_data = self.layer2_datahandler.df_test_data
        # X_test = df_test_data[x_vars]
        # y_test = df_test_data[y_var]

        if self.model_name == 'OLS' or self.model_name == 'Linear Regression':
            model = LinearRegression().fit(X, y)
        elif self.model_name == 'Ridge':
            model = Ridge(**self.kwargs).fit(X, y)
        elif self.model_name == 'Lasso':
            model = Lasso(**self.kwargs).fit(X, y)
        elif self.model_name == 'Gradient Boosting':
            model = GradientBoostingRegressor(**self.kwargs).fit(X, y)
        elif self.model_name == 'SARIMAX':
            exog_vars = df_data[[c for c in df_data.columns if "p_trump_win" not in c]]
            model = SARIMAX(y, exog = exog_vars, order = (1, 1, 1), seasonal_order = (0, 0, 0, 0)).fit()
            # print(model.summary())
            # y_pred = model.predict(start=len(y), end=len(y) + len(y_test) - 1, exog = X_test[['other', 'trump', 'other_sentiment_indic', 'trump_sentiment_indic']])
            # predictions = pd.DataFrame({'y_test': y_test.values, 'y_pred': y_pred.values})
            # print(predictions.head())
            # print(f"Modelname: {self.model_type}, Sentiment Model: {self.layer2_datahandler.sentiment_model}, rmse: {mean_squared_error(y_test.values, y_pred.values)}")
        else:
            raise ValueError(f'Invalid model_name: {self.model_name}')
        
        if self.model_type == 'SARIMAX':
            prediction = model.predict(start=0, end=len(y) - 1, exog = X[['other', 'trump', 'other_sentiment_indic', 'trump_sentiment_indic']]).values()
        else:
            prediction = model.predict(X)

        accuracy, mae, rmse, r2, mape = self.evaluate_model(y, prediction)

        if self.verbose:
            # print(f'Modelname: {self.model_name}, Sentiment Model: {self.layer2_datahandler.sentiment_model}, kwargs: {self.kwargs}, r2: {r2_score(y, model.predict(X))}')
            print(f'Modelname: {self.model_name}, Sentiment Model: {self.layer2_datahandler.sentiment_model}, kwargs: {self.kwargs}, accuracy: {accuracy}, r2: {r2}, mape: {mape}, mae: {mae}, rmse: {rmse}')
            print(f"Actual Counts: {np.unique(np.where(y > 0.5, 1, 0), return_counts = True)}, Predicted Counts: {np.unique(np.where(prediction > 0.5, 1, 0), return_counts = True)}")
        return model
    
    def evaluate_model(self, y_true, y_pred):
        # accrate means same result prediction: 0 - 0.5, 0.5 - 1
        binary_y_pred = np.where(y_pred > 0.5, 1, 0)
        binary_y_true = np.where(y_true > 0.5, 1, 0)
        accuracy = accuracy_score(binary_y_true, binary_y_pred)
        
        # calculate accuracy for each class (other=less than 0.5, trump=greater than 0.5)
        actual_classes = np.where(y_true < 0.5, 1, 2)
        predicted_classes = np.where(y_pred < 0.5, 1, 2)

        # Masks for each class in y_actual
        class1_mask = (actual_classes == 1)
        class2_mask = (actual_classes == 2)

        # Accuracy for Class 1
        class1_correct = np.sum(predicted_classes[class1_mask] == actual_classes[class1_mask])
        class1_total = np.sum(class1_mask)
        class1_accuracy = class1_correct / class1_total if class1_total > 0 else 0

        # Accuracy for Class 2
        class2_correct = np.sum(predicted_classes[class2_mask] == actual_classes[class2_mask])
        class2_total = np.sum(class2_mask)
        class2_accuracy = class2_correct / class2_total if class2_total > 0 else 0

        # Print results
        print(f"Other Accuracy: {class1_accuracy:.2f}")
        print(f"Trump Accuracy: {class2_accuracy:.2f}")
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return accuracy, mae, rmse, r2, mape



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
    prob_modeler_ols = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='OLS')
    # prob_modeler_sarimax = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='SARIMAX')
    prob_modeler_ridge1 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='Ridge', alpha=0.1)
    # prob_modeler_lasso01 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_type='Lasso', alpha=0.1)
    prob_modeler_gb1 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='Gradient Boosting', n_estimators=50, max_depth=2)
    prob_modeler_gb2 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='Gradient Boosting', n_estimators=20, max_depth=2)
    prob_modeler_gb3 = EnhancedProbabilityModeler(layer2_datahandler=layer2_datahandler, model_name='Gradient Boosting', n_estimators=100, max_depth=2)
    
    model_pipeline = [
        # prob_modeler_sarimax,
        prob_modeler_ols, 
        prob_modeler_ridge1,
        # prob_modeler_lasso01, 
        prob_modeler_gb1,
        prob_modeler_gb2,
        prob_modeler_gb3
    ]

    for modeler in model_pipeline:
        print("--------------------------------RUNNING MODEL--------------------------------")
        model = modeler.run_model()