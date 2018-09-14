
# Basics
import numpy as np
import pandas as pd
import sklearn as sk
from time import time
import random
import pickle
from copy import copy, deepcopy

# Sklean Models
from models import regression_options, classification_options

# Training prep/scoring metrics
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score as r2

# Model Persistance
from sklearn.externals import joblib

# Plotting
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout



class StackML:

    '''
    Parameters:
    ----------------------------

    cv_folds: int, default 3
    'cv_folds' determines the number of CV folds Kfold or
       TimeSeriesSplit will generate. It should be an int

    'reuse_features' should be a string or a list of strings with the
       name of columns that you want to add to the combiner model's
       training set.

    'shuffle' determined whether Kfold will shuffle the data before
       generating CV indices. It should be True or False.

    'timeseries' determines wich CV folding algorithm to use.
       True --> TimeSeriesSplit, False --> Kfold

    'base_models' can be 'all' or a list of models to use as the base
       models in the ensemble.

    'combiner' should be a string

    Returns:
    ----------------------------

    '''

    def __init__(self,

                cv_folds = 3,
                shuffle = False,
                verbose = False,
                timeseries = False,
                base_models = 'all',
                use_best_model = True,
                reuse_features = None,
                combiner = 'MLPRegressor',
                discard_overfitting_base_models = True,

                ):

        self.verbose = verbose
        self.discard = discard_overfitting_base_models

        # Kfold index generator parameters
        self.shuffle = shuffle
        self.cv_folds = cv_folds
        self.cv_indices = {'train':[], 'test':[]}
        self.timeseries = timeseries

        model_options = copy(regression_options)

        for model in model_options:
            model_options[model]['info'] = {'RMSE_train':[], 'RMSE_test':[]}

        if base_models == 'all':
            self.base_models = model_options
        else:
            base_models = [base_models]

        if isinstance(combiner, str):
            if combiner not in model_options:
                raise KeyError(f'''
                    The combiner model should either be a dict with a
                    callable model or one of the following string options:
                    {model_options.keys()}
                ''')
            else:
                self.combiner = {
                    'model' : copy(model_options[combiner]['model']),
                    'name' : model_options[combiner]['name'] + '_combiner',
                    'info' : {'RMSE_train':[], 'RMSE_test':[]}
                }
        elif isinstance(combiner, dict):
            self.combiner = combiner
        else:
            self.combiner = None

        self.reuse_features = reuse_features


    def fit(self, Xtrain, ytrain):

        self.Xtrain = Xtrain
        self.ytrain = ytrain

        # Initialize cross validation index generator
        if self.timeseries:
            cv_splitter = TimeSeriesSplit(n_splits = self.cv_folds)
        else:
            cv_splitter = KFold(n_splits = self.cv_folds,
                                 shuffle = self.shuffle,
                                 random_state = 1)


        def _CV_stats(model_dict, CV_Xtrain, CV_Xtest,
                                  CV_Ytrain, CV_Ytest):

            model = model_dict['model']
            train_score = round(model.score(CV_Xtrain, CV_Ytrain), 4)
            test_score = round(model.score(CV_Xtest, CV_Ytest), 4)
            model_dict['info']['RMSE_train'].append(train_score)
            model_dict['info']['RMSE_test'].append(test_score)

            if self.verbose:
                msg = f'Train score: {train_score} || Test score: {test_score}'
                print(msg, '\n')


        def _train_base_models():

            if self.verbose:
                print('\nTraining Base Models')
                print('-------------------------------------------------')


            iteration = 1
            for train_inds, test_inds in cv_splitter.split(self.Xtrain):

                if self.verbose:
                    print(f'\nCV fold: {iteration}')
                    print('---------------------------')

                # Log CV index info
                self.cv_indices['train'].append(train_inds)
                self.cv_indices['test'].append(test_inds)

                # Slice CV sets
                CV_Xtrain = self.Xtrain.iloc[train_inds, :]
                CV_Ytrain = self.ytrain.iloc[train_inds]
                CV_Xtest = self.Xtrain.iloc[test_inds, :]
                CV_Ytest = self.ytrain.iloc[test_inds]

                # Train, save CV performance stats
                for model_name, model_dict in self.base_models.items():

                    t = time()

                    model_dict['model'].fit(CV_Xtrain, CV_Ytrain)

                    if self.verbose:
                        print(f'{model_name}: {round(time() - t, 4)}s')

                    _CV_stats(model_dict, CV_Xtrain, CV_Xtest,
                                          CV_Ytrain, CV_Ytest)

                iteration += 1

                if self.discard:
                    for model in list(self.base_models.keys()):
                        score = self.base_models[model]['info']['RMSE_test'][-1]
                        if score < .60:
                            self.base_models.pop(model)
                            if self.verbose:
                                print(f'---- Dropped {model} ----')



        def _train_combiner():
        # Compile Predictions from base models into features for
        # training the Neural Network

            # Restarting index generator
            if self.timeseries:
                cv_splitter = TimeSeriesSplit(n_splits = self.cv_folds)
            else:
                cv_splitter = KFold(n_splits = self.cv_folds,
                                     shuffle = self.shuffle,
                                     random_state = 1)

            # Obtain combiner model features using predictions from base models
            self.combiner_features = self._ensemble_features(self.Xtrain)

            if self.verbose:
                print('\nTraining Combiner')
                print('-------------------------------------------------')

            # Train combiner with base model predictions
            iteration = 1
            for train_inds, test_inds in cv_splitter.split(self.combiner_features):

                model_name = self.combiner['name']

                CV_Xtrain = self.combiner_features.iloc[train_inds, :]
                CV_Xtest = self.combiner_features.iloc[test_inds, :]
                CV_Ytrain = self.ytrain.iloc[train_inds]
                CV_Ytest = self.ytrain.iloc[test_inds]

                if self.verbose:
                    print(f'CV fold {iteration}')

                t = time()

                # Train, save CV performance stats
                self.combiner['model'].fit(CV_Xtrain, CV_Ytrain)

                if self.verbose:
                    print(f'{model_name}: {round(time() - t, 4)}s')

                _CV_stats(self.combiner, CV_Xtrain, CV_Xtest,
                                         CV_Ytrain, CV_Ytest)

                iteration+=1



        _train_base_models()
        if self.combiner:
            _train_combiner()


    def _ensemble_features(self, data):
    # Given a set of base model predictions, return features for the combiner model

        combiner_features = pd.DataFrame(columns = self.base_models.keys())
        it = 0
        for model in self.base_models.keys():
            tempPred = self.base_models[model]['model'].predict(data)
            combiner_features[model] = tempPred
            it+=1

        if self.reuse_features:
            tempData = data[self.reuse_features].reset_index(drop = True)
            combiner_features = pd.concat([combiner_features, tempData], axis = 1)

        return combiner_features


    def score(self, Xtest, ytest, model = 'combiner'):
    # Return R2 score of model on test set

        self.Xtest = Xtest
        self.ytest = ytest

        assert self.ytest is not None, \
            'Tried to score model with saved test data, but no test data was found'

        if len(self.base_models) == 1 and self.combiner == None:
            mod = list(self.base_models.keys())[0]
            return self.base_models[mod]['model'].score(self.Xtest, self.ytest)

        elif model == 'combiner':
            assert self.combiner, \
                'Attempted to score with combiner model but no combiner model was found'
            self.combiner_features = self._ensemble_features(self.Xtest)
            return self.combiner['model'].score(self.combiner_features, self.ytest)

        else:
            assert model in self.base_models.keys(), \
                'Attempted to score {} but {} was not found in base models'.format(model)
            return self.base_models[model]['model'].score(self.Xtest, self.ytest)


    def predict(self, X, model = 'combiner'):
    # Given X, obtain combiner features from base model predictions,
    # then return prediction from combiner

        if len(self.base_models) == 1 and self.combiner == None:
            return self.base_models[list(self.base_models.keys())[0]]['model'].predict(X)

        elif model == 'combiner':
            assert self.combiner, \
                'Attempted to make a prediction with combiner model but no combiner model was found'
            self.combiner_features = self._ensemble_features(X)
            return self.combiner['model'].predict(self.combiner_features)

        else:
             assert model in self.base_models.keys(), \
                'Attempted to predict with {} but {} was not found in base models'.format(model)
             return self.base_models[model]['model'].predict(X)


    def plot_prediction(self,
                        filename = None,
                        auto_open = False,
                        data = None,
                        x_axis_data = None):

        '''
        plot_prediction will automatically generate a plot of model predictions
        vs. actual values using self.Xtest and self.ytest, or, it can take a
        dictionary of prediction data and generate a custom plot. It will output
        an .html file at a given directory/filename, and automatically open a
        browser and display the plot if auto_open == True.


        - 'filename' should be a string with the name of the file, without file
            extension, in the form 'dir/dir.../filename'

        - x_axis should be the name of the column to use as

        - 'auto_open' toggles whether to open the browser

        - 'data' should at a minimum be a dictionary in the form of:

                {'predictions':<pd.Series>, 'actual':<pd.Series>}

            or with optional metadata:

                {'predictions':<pd.Series>, 'predictions_name':'<somestring>',
                 'actual':<pd.Series>,      'actual_name':'<somestring>',
                 'y_label':'<somestring>',  'x_label':'<somestring>',
                 'x_data': <pd.Series>,     'plot_title':'<somestring>', }

        '''


        predicted_price = self.predict(self.Xtest)
        actual_price = self.ytest

        if x_axis_data:
            if type(x_axis_data) == str:
                x_data = self.Xtest[x_axis_data]
            else:
                assert len(x_axis_data) == len(self.Xtrain.index), \
                        "lengths of x_axis_data and training data are mismatched"
                x_data = x_axis_data
        else:
            x_data = None


        Actual = Scatter(
            x = x_data,
            y = actual_price,
            name = 'Actual Values'
            )

        Prediction = Scatter(
            x = x_data,
            y = predicted_price,
            name = 'Model Predictions'
            )

        layout = go.Layout(
            title = 'Model Predictions vs. Actual'
        )

        if filename:
            filename = f'./plots/{filename}.html'
        else:
            filename = './plots/plot.html'

        plotData = [Actual, Prediction]
        plotly.offline.plot({'data': plotData,
                             'layout': layout},
                              filename = filename,
                              auto_open = auto_open
                              )



    def save_model(self, path = './model/saved_models/', f_name = 'saved_model'):

        # Don't save data with model if selected
        if not self.keep_data:
            self.Xtrain = None
            self.ytrain = None
            self.Xtest = None
            self.ytest = None

        with open(path + f_name + '.sav', 'wb') as fo:
            pickle.dump(self, fo)


    def export_data(self, f_name, path = './model/saved_data/{}'):

        if f_name[len(path)-4:] != '.sav':
            f_name = f_name + '.sav'

        data = {'Xtrain': self.Xtrain,
                'Ytrain': self.ytrain,
                'Xtest': self.Xtest,
                'Ytest': self.ytest}

        loc = path.format(f_name)
        with open(loc, 'wb') as fo:
            pickle.dump(data, fo)


def load_model(path):
    # loaded_model = joblib.load(path)
    with open(path, 'rb') as fo:
        loaded_model = pickle.load(fo)
    return loaded_model
