
# Basics
import numpy as np
import pandas as pd
import sklearn as sk
import time
import random
import pickle

# Machine Learning
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, BaggingRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import (BayesianRidge, SGDRegressor,
                                  LinearRegression, Lasso, ElasticNet)

# Training prep/scoring metrics
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score as r2

# Model Persistance
from sklearn.externals import joblib

# Plotting
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout


#TODO Replace all normal indexing with .iloc to allow for any type of index
#TODO Add algorithm that removes severely underperforming base
#     before using predictions for training combiner
#TODO Improve plotting functionality

class Ensemble:

    # TODO:190 Write a better high-level description of the model
    '''
     --- Great description of model ---
         - 'Xtrain', 'Ytrain', 'Xtest', and 'Ytest' should be pandas
            DataFrames filled with only numerical data.
                - 'Xtrain' and 'Xtest' can have any number of columns, 'Ytrain'
                   and 'Ytest' should be a single column

         - 'cv_folds' determines the number of CV folds Kfold or
            TimeSeriesSplit will generate. It should be an int

         - 'reuse_features' should be a string or a list of strings with the
            name of columns that you want to add to the combiner model's
            training set.

         - 'shuffle' determined whether Kfold will shuffle the data before
            generating CV indices. It should be True or False.

         - 'timeseries' determines wich CV folding algorithm to use.
            True --> TimeSeriesSplit, False --> Kfold

         - 'base_models' can be 'all' or a list of models to use as the base
            models in the ensemble.

         - 'combiner' should be a string
    '''
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest,

                cv_folds = 3,
                reuse_features = None,
                shuffle = False,
                timeseries = True,
                base_models = 'all',
                combiner = 'MLPRegressor',
                keep_data = True,
                discard_overfitting_base_models = True,

                ):

        assert type(Xtrain) == pd.core.frame.DataFrame, \
            'Data must be in the form of a pandas DataFrame'

        # Data
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.features = Xtrain.columns
        self.keep_data = keep_data
        self.discard = discard_overfitting_base_models

        # If training data was normalized, it's recommended to save params
        self.minX = None
        self.maxX = None
        self.meanX = None
        self.norm_option = None

        # If model predicts a future time in a time-series, save deltaT
        # deltaT should be a datetime.timedelta obj
        self.deltaT = None

        # Kfold index generator parameters
        self.cv_folds = cv_folds
        self.cv_indices = [[],[]]
        self.shuffle = shuffle
        self.timeseries = timeseries


        # Initialize models using 'all' or user-defined selection
        model_options = {'MLPRegressor': \
                            {'model':MLPRegressor(learning_rate = 'adaptive',
                                                  max_iter=500,
                                                  learning_rate_init=.005),
                             'name':'MLP NN'},

                        'RandomForestRegressor': \
                            {'model':RandomForestRegressor(n_estimators = 20,
                                                           max_features = 2),
                             'name':'Random Forest'},

                        'BayesianRidge': \
                            {'model':BayesianRidge(),
                             'name':'Bayesian Ridge'},

                        'Lasso': \
                            {'model':Lasso(),
                             'name':'Lasso Regressor'},

                        'GradientBoostingRegressor': \
                            {'model':GradientBoostingRegressor(max_features=2),
                             'name':'Gradient Boost'},

                        'ElasticNet': \
                            {'model':ElasticNet(selection='random'),
                             'name':'Elastic Net Regressor'},

                        'KernelRidge': \
                            {'model':KernelRidge(),
                             'name':'Kernel Ridge'},

                        'SVR': \
                            {'model':SVR(),
                             'name':'SVR'},

                        'BaggingRegressor': \
                            {'model':BaggingRegressor(),#base_estimator = LinearRegression()),
                             'name':'Bagging Regressor'},

                        'ExtraTreesRegressor': \
                            {'model':ExtraTreesRegressor(),
                             'name':'Extra Trees Regressor'},

                        'KNeighborsRegressor': \
                            {'model':KNeighborsRegressor(),
                             'name':'K Neighbors Regressor'},

                        'RadiusNeighborsRegressor': \
                            {'model':RadiusNeighborsRegressor(),
                             'name':'Radius Neighbors Regressor'},

                        'DecisionTreeRegressor': \
                            {'model':DecisionTreeRegressor(),
                             'name':'Decision Tree Regressor'},

                        'AdaBoostRegressor': \
                            {'model':AdaBoostRegressor(),#base_estimator = LinearRegression()),
                             'name':'AdaBoost'},

                        'LinearRegression': \
                            {'model':LinearRegression(),
                             'name':'Linear Regression'},

                        'SGDRegressor': \
                            {'model':SGDRegressor(max_iter=1000),
                             'name':'Stochastic Gradient Descent'},
                    }

        for model in model_options:
            model_options[model]['info'] = {'RMSE_train':[],
                                            'RMSE_test':[]}

        if type(base_models) == str:
            if base_models == 'all':

                self.base_models = {}
                # HACK Exclude models that aren't performing well out of the box
                for model in model_options.keys():
                    if model != 'MLPRegressor' \
                    and model != 'KernelRidge' \
                    and model != 'SVR' \
                    and model != 'RadiusNeighborsRegressor' \
                    and model != 'RandomForestRegressor':# \
                    # and model != 'ElasticNet' \
                    # and model != 'KNeighborsRegressor' \
                    # and model != 'DecisionTreeRegressor' \
                    # and model != 'GradientBoostingRegressor' \
                    # and model != 'ExtraTreesRegressor' \
		            # and model != 'AdaBoostRegressor':

                        self.base_models[model] = model_options[model]

            else:
                base_models = [base_models]

                assert [model for model in base_models if model in model_options.keys()], \
                        "'models' must be a list containing valid model names"

                self.base_models = {}
                for model in base_models:
                    self.base_models[model] = model_options[model]

        if combiner:
            self.combiner = model_options[combiner]
        else:
            self.combiner = None
            assert len(self.base_models) == 1, \
            'Currently only a single base model is supported if no combiner model is used'



        # Parameter used in ensemble_features: features to use a second
        # time when training the combiner model
        self.reuse_features = reuse_features

    def train(self):

        # Initialize cross validation index generator
        if self.timeseries:
            _cv_splitter = TimeSeriesSplit(n_splits = self.cv_folds)
        else:
            _cv_splitter = KFold(n_splits = self.cv_folds,
                                 shuffle = self.shuffle,
                                 random_state = 1)


        def _info(train_score, test_score, iteration):
        # Save model performance metrics from each training iteration

            info = 'R2 in I{}: Train: {}, Test: {}'.format(iteration,
                                                           train_score,
                                                           test_score)
            return [info]


        def _CV_performance(model, CV_Xtrain, CV_Xtest,
                                   CV_Ytrain, CV_Ytest):

            train_score = round(model.score(CV_Xtrain, CV_Ytrain), 4)
            test_score = round(model.score(CV_Xtest, CV_Ytest), 4)

            return train_score, test_score



        def _train_base_models():

            iteration = 1
            for train_index, test_index in _cv_splitter.split(self.Xtrain):
                print('CV fold: %d' %(iteration))
                print('----------------------------------------------------')

                # Log CV index info
                self.cv_indices[0]. \
                    append('Training indices: iteration {}'.format(iteration))
                self.cv_indices[1].append(train_index)

                self.cv_indices[0]. \
                    append('Test indices: iteration {}'.format(iteration))
                self.cv_indices[1].append(test_index)

                CV_Xtrain = self.Xtrain.iloc[train_index, :]
                CV_Ytrain = self.Ytrain.iloc[train_index]

                CV_Xtest = self.Xtrain.iloc[test_index, :]
                CV_Ytest = self.Ytrain.iloc[test_index]

                models = list(self.base_models.keys())
                for model in models:

                    t = time.time()
                    mod = self.base_models[model]

                    mod['model'].fit(CV_Xtrain, CV_Ytrain)

                    train_score, test_score = \
                        _CV_performance(mod['model'], CV_Xtrain, CV_Xtest,
                                                      CV_Ytrain, CV_Ytest)

                    mod['info']['RMSE_train'].append(train_score)
                    mod['info']['RMSE_test'].append(test_score)

                    self.base_models[model] = mod
                    elapsed = round(time.time() - t, 4)


                    training_time = \
                        'Trained {} in {}s'.format(mod['name'], elapsed)

                    scores = \
                        'Train score: {} || Test score {}'.format(train_score,
                                                                  test_score)

                    if self.discard:
                        if test_score < .60:
                            self.base_models.pop(model)
                            training_time += '      **Removed**'

                    print(training_time)
                    print(scores)

                    print()

                iteration += 1


        def _train_combiner():
        # Compile Predictions from base models into features for
        # training the Neural Network

            # Restarting Kfold index generator
            if self.timeseries:
                _cv_splitter = TimeSeriesSplit(n_splits = self.cv_folds)
            else:
                _cv_splitter = KFold(n_splits = self.cv_folds,
                                     shuffle = self.shuffle,
                                     random_state = 1)

            # Obtain combiner model features using predictions from base models
            self.combiner_features = self._ensemble_features(self.Xtrain)

            # Train Neural Network with base model predictions
            iteration = 1
            for train_index, test_index in _cv_splitter.split(self.combiner_features):

                model_name = self.combiner['name']
                CV_Xtrain = self.combiner_features.iloc[train_index, :]
                CV_Xtest = self.combiner_features.iloc[test_index, :]

                CV_Ytrain = self.Ytrain.iloc[train_index]
                CV_Ytest = self.Ytrain.iloc[test_index]

                print('Combiner CV fold %d' %(iteration))

                t = time.time()
                self.combiner['model'].fit(CV_Xtrain,
                                           CV_Ytrain)

                train_score, test_score = \
                    _CV_performance(self.combiner['model'], CV_Xtrain, CV_Xtest,
                                                            CV_Ytrain, CV_Ytest)

                self.combiner['info']['RMSE_train'].append(train_score)
                self.combiner['info']['RMSE_test'].append(test_score)

                scores = \
                    'Train score: {} || Test score {}'.format(train_score,
                                                              test_score)

                elapsed = round(time.time()-t, 4)
                print('Trained {} in {}s'.format(model_name, elapsed))
                print(scores)

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


    # TODO Assertions should probably all be done in __init__
    def score(self, model = 'combiner'):
    # Return R2 score of model on test set

        assert self.Ytest is not None, \
            'Tried to score model with saved test data, but no test data was found'

        if len(self.base_models) == 1 and self.combiner == None:
            mod = list(self.base_models.keys())[0]
            return self.base_models[mod]['model'].score(self.Xtest, self.Ytest)

        elif model == 'combiner':
            assert self.combiner, \
                'Attempted to score with combiner model but no combiner model was found'
            self.combiner_features = self._ensemble_features(self.Xtest)
            return self.combiner['model'].score(self.combiner_features, self.Ytest)

        else:
            assert model in self.base_models.keys(), \
                'Attempted to score {} but {} was not found in base models'.format(model)
            return self.base_models[model]['model'].score(self.Xtest, self.Ytest)


    # TODO Assertions should probably all be done in __init__
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


    def plot_prediction(self, filename,
                        auto_open = False,
                        data = None,
                        x_axis_data = None):

        '''
        plot_prediction will automatically generate a plot of model predictions
        vs. actual values using self.Xtest and self.Ytest, or, it can take a
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

        if data:

            assert 'predictions' and 'actual' in data.keys(), \
                    "'data' requires 'predictions' and 'actual' keys"


            pred_name = data['predictions_name']            \
                        if 'predictions_name' in data.keys() \
                        else 'Model Predictions'

            x_data = data['x_data']             \
                     if 'x_data' in data.keys()  \
                     else None

            act_name = data['actual_name']              \
                       if 'actual_name' in data.keys()   \
                       else 'Actual Values'

            y_label = data['y_label']              \
                      if 'y_label' in data.keys()   \
                      else None

            x_label = data['x_label']               \
                      if 'x_label' in data.keys()    \
                      else None

            plot_title = data['plot_title']             \
                         if 'plot_title' in data.keys()  \
                         else 'Actual Values vs. Model Predictions'



            Actual = Scatter(
                x = x_data,
                y = data['actual'],
                name = act_name
            )

            Prediction = Scatter(
                x = x_data,
                y = data['predictions'],
                name = pred_name
            )

            x_axis_template = dict(title = x_label)
            y_axis_template = dict(title = y_label)

            layout = go.Layout(
                xaxis = x_axis_template,
                yaxis = y_axis_template,
                title = plot_title
            )


        else:

            actual_price = self.Ytest
            predicted_price = self.predict(self.Xtest)

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



        filename = filename + '.html'
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
            self.Ytrain = None
            self.Xtest = None
            self.Ytest = None

        with open(path + f_name + '.sav', 'wb') as fo:
            pickle.dump(self, fo)


    def export_data(self, f_name, path = './model/saved_data/{}'):

        if f_name[len(path)-4:] != '.sav':
            f_name = f_name + '.sav'

        data = {'Xtrain': self.Xtrain,
                'Ytrain': self.Ytrain,
                'Xtest': self.Xtest,
                'Ytest': self.Ytest}

        loc = path.format(f_name)
        with open(loc, 'wb') as fo:
            pickle.dump(data, fo)


def load_model(path):
    # loaded_model = joblib.load(path)
    with open(path, 'rb') as fo:
        loaded_model = pickle.load(fo)
    return loaded_model


class features:
# class 'features' generates a formatted training and test set with the option to
# generates staggered target variable columns for training a model to
# predict <y_gen> time steps into the future.

    def __init__(self, data,
                 target_variable = 'price',
                 split_percent = .8,
                 y_gen = 0,
                 keep_all = True,
                 reset_index = False):

        '''
        - 'data' should be a pandas DataFrame of numerical data with columns for
            each feature, including the target variable.

        - target_variable should be a string with the name of the column that contains
            the target variable

        - split_percent should be a decimal with the percentage of data the model
            should train on. (ex. 90 percent train, 10 percent test --> split_percent = .9)

        - y_gen should be the number of time steps to generate y to. y_gen = 3
            will produce 4 columns; col[0] being y, col[1:n] being (y + 1 timestep)...(y + n timesteps)

        - keep_all toggles whether we want to keep columns between y and (y + n timesteps), or just
            keep y and (y + n timesteps). In other words, col[0] and col[n], or col[0:n]

        '''


        assert 0 < split_percent < 1, 'split percent must be a decible between 0 and 1'
        assert type(data) == pd.core.frame.DataFrame, 'data must be a pandas dataframe'

        # Check to be sure that 'data' only contains numerical data that can be trained on
        for column in data.columns:
            assert (type(data[column][0]) in [int, float, np.float64]), \
                    'Columns must only contain numerical data. Columns in question: {}'.format(column)


        self.target_variable = target_variable
        self.split_percent = split_percent
        self.y_gen = y_gen
        self.keep_all = keep_all
        self.reset_index = reset_index

        # Generate target data columns
        self.data = self.__generate_y(data)

        # Generate the rounded index for splitting the dataset
        self.split_ind = self.__split(self.split_percent)

        # Split data into X and Y
        self.X = self.data.iloc[:, :(self.data.shape[1]-(self.y_gen+1))]
        if self.y_gen > 0:
            self.Y = self.data.iloc[:, (self.data.shape[1]-(self.y_gen+1)):]
        else:
            self.Y = self.data[target_variable]

        # Split Y into train and test sets
        if self.Y.shape == (self.Y.shape[0],):
            self.Ytrain = self.Y[:self.split_ind]
            self.Ytest = self.Y[self.split_ind:]
        else:
            self.Ytrain = self.Y.iloc[:self.split_ind, :]
            self.Ytest = self.Y.iloc[self.split_ind:, :]

        # Split X into train and test sets
        self.Xtrain = self.X.iloc[:self.split_ind, :]
        self.Xtest = self.X.iloc[self.split_ind:, :]



    def __split(self, split_percent):
        length = self.data.shape[0]
        split_ind = round(length*split_percent)
        return split_ind


    def __generate_y(self, data):
    # Generate Y data for n number of time steps beyond 'current'

        if self.y_gen == 0:
            if self.reset_index:
                print('Despite index not being affected by y_gen, reseting index')

            return data.reset_index(drop=True)

        if self.reset_index:
            data = data.reset_index(drop = True)

        tempY = list(data[self.target_variable])
        size = len(tempY)
        Y = pd.DataFrame()
        for i in range(self.y_gen + 1):
            if i == 0:
                Y['current'] = np.ones(size)
            else:
                Y['%d_dt_future' %(i)] = np.ones(size)

        for i in range(self.y_gen + 1):
            Y.iloc[:size-i, i] = tempY[i:size+1]

        data = pd.concat([data,Y], axis = 1)
        data = data.drop([i for i in range(len(Y)-self.y_gen, len(Y))])
        data = data.drop(self.target_variable, axis = 1)

        return data


    def ret(self):

        if self.keep_all:
            return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest
        else:
            if self.y_gen == 0:
                return self.Xtrain, self.Ytrain, self.Xtest, self.Ytest
            else:
                return self.Xtrain, self.Ytrain.iloc[:, self.y_gen], \
                       self.Xtest,  self.Ytest.iloc[:,  self.y_gen]





class optimize:
    def __init__(self, ensemble_object):

        # Get attributes from ensemble
        self.ensemble =         ensemble_object
        self.base_models =      self.ensemble.base_models
        self.combiner =         self.ensemble.combiner
        self.Xtrain =           self.ensemble.Xtrain
        self.Ytrain =           self.ensemble.Ytrain
        self.cv_folds =         self.ensemble.cv_folds
        self.cv_indices =       self.ensemble.cv_indices
        self.shuffle =          self.ensemble.shuffle
        self.timeseries =       self.ensemble.timeseries
        self.shuffle =          self.ensemble.shuffle

        # Initialize cv and model dictionary
        self.models = self.get_models()
        self.results = {model:{'score':None, 'params':None} for model in self.models.keys()}
        self.cv = self.init_cv()



    def init_cv(self):

        if self.timeseries:
            cv = TimeSeriesSplit(n_splits = self.cv_folds)
        else:
            cv = Kfold(n_splits = self.cv_folds,
                       shuffle = self.shuffle,
                       random_state = 1)
        return cv


    def search(self):
        for model in self.models.keys():

            gsearch = GridSearchCV(estimator = self.models[model]['model'],
                                   param_grid = self.models[model]['param_grid'],
                                   scoring = 'neg_mean_squared_error',
                                   cv = self.cv,
                                   n_jobs = 1)

            print('Performing search for {}'.format(model))
            print('--------------------------------------------------------')
            results = gsearch.fit(self.Xtrain, self.Ytrain)

            print('Best score for {}: {}'.format(model, results.best_score_))
            self.results[model]['score'] = results.best_score_
            self.results[model]['params'] = results.best_params_


    def get_models(self):

        model_options = {'MLPRegressor': \
                            {'model':MLPRegressor(),
                             'name':'MLP NN',
                             'param_grid' : {'hidden_layer_sizes' : [(100, ), (100, 2), (100, 3), (200, )],
                                          'activation' : ['relu', 'logistic', 'tanh', 'identity'],
                                          'solver' : ['adam', 'lbfgs', 'sgd'],
                                          'alpha' : [0.0001, 0.00001, 0.0005, 0.001],
                                          'batch_size' : ['auto', 'lbfgs'],
                                          'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                                          'learning_rate_init' : [0.001, 0.0001, 0.005, 0.01],
                                          'power_t' : [0.5, 2, 1, .2],
                                          'max_iter' : [100, 200, 500, 1000],
                                          'shuffle' : [True, False],
                                          'random_state' : [None],
                                          'tol' : [0.0001, 0.00001, 0.001],
                                          'verbose' : [True, False],
                                          'warm_start' : [True, False],
                                          'momentum' : [0.1, 0.3, 0.7, 0.9],
                                          'nesterovs_momentum' : [True, False],
                                          'early_stopping' : [True, False],
                                          'validation_fraction' : [0.1, 0.2, 0.3],
                                          'beta_1' : [0.5, 0.7, 0.9],
                                          'beta_2' : [0.999, 0.7, 0.5],
                                          'epsilon' : [1e-08, 1e-07, 1e-09]
                                          }
                             },

                        'RandomForestRegressor': \
                            {'model':RandomForestRegressor(),
                             'name':'Random Forest',
                             'param_grid' : {'n_estimators' : [5, 40, 100, 200, 300],
                                          'criterion' : ['mse', 'mae'],
                                          'max_depth' : [None, 2, 4, 6],
                                          'min_samples_split' : [2, 4, 10],
                                          'min_samples_leaf' : [1],
                                          'min_weight_fraction_leaf' : [0.0],
                                          'max_features' : ['auto', 'sqrt', 'log2'],
                                          'max_leaf_nodes' : [None, 4, 20, 100],
                                          'min_impurity_decrease' : [0.0],
                                          'min_impurity_split' : [None],
                                          'bootstrap' : [True, False],
                                          'oob_score' : [True, False],
                                          'n_jobs' : [1, -1],
                                          'random_state' : [None],
                                          'verbose' : [0],
                                          'warm_start' : [True, False]
                                          }
                             },

                        'BayesianRidge': \
                            {'model':BayesianRidge(),
                             'name':'Bayesian Ridge',
                             'param_grid' : {'n_iter' : [200, 300, 500, 1000],
                                          'tol' : [1e-3, 1e-2, 1e-4],
                                          'alpha_1' : [1e-06, 1e-05, 1e-07],
                                          'alpha_2' : [1e-06, 1e-07, 1e-05],
                                          'lambda_1' : [1e-06, 1e-07, 1e-05],
                                          'lambda_2' : [1e-06, 1e-07, 1e-05],
                                          'compute_score' : [True, False],
                                          'fit_intercept' : [True, False],
                                          'normalize' : [True, False],
                                          'copy_X' : [True],
                                          'verbose' : [True, False]
                                          }
                             },

                        'Lasso': \
                            {'model':Lasso(),
                             'name':'Lasso Regressor',
                             'param_grid' : {'alpha' : [1.0, 2.0, 1.5, 2.5],
                                          'fit_intercept' : [True, False],
                                          'normalize' : [True, False],
                                          'precompute' : [True, False],
                                          'copy_X' : [True],
                                          'max_iter' : [500, 1000, 2000],
                                          'tol' : [0.0001, 0.001, 0.00001, 0.0005],
                                          'warm_start' : [True, False],
                                          'positive' : [True, False],
                                          'random_state' : [None],
                                          'selection' : ['cyclic', 'random']
                                          }
                             },

                        'GradientBoostingRegressor': \
                            {'model':GradientBoostingRegressor(),
                             'name':'Gradient Boost',
                             'param_grid' : {'loss' : ['ls', 'lad', 'huber', 'quantile'],
                                          'learning_rate' : [0.1, 0.05, .15, .2],
                                          'n_estimators' : [50, 100, 200, 300, 400],
                                          'subsample' : [0.75, 0.05, 1.0],
                                          'criterion' : ['friedman_mse', 'mse', 'mae'],
                                          'min_samples_split' : [2],
                                          'min_samples_leaf' : [1],
                                          'min_weight_fraction_leaf' : [0.0],
                                          'max_depth' : [3],
                                          'min_impurity_decrease' : [0.0],
                                          'min_impurity_split' : [None],
                                          'init' : [None],
                                          'random_state' : [None],
                                          'max_features' : [None, 'sqrt', 'log2'],
                                          'alpha' : [0.9],
                                          'verbose' : [0],
                                          'max_leaf_nodes' : [None],
                                          'warm_start' : [True, False],
                                          'presort' : ['auto']
                                          }
                             },

                        'ElasticNet': \
                            {'model':ElasticNet(),
                             'name':'Elastic Net Regressor',
                             'param_grid' : {'alpha' : [0.5, 0.7, 1.0, 1.2],
                                          'l1_ratio' : [.1, .5, .7, .9, .95, .99, 1],
                                          'fit_intercept' : [True, False],
                                          'normalize' : [True, False],
                                          'precompute' : [True, False],
                                          'max_iter' : [500, 1000, 1500, 2000],
                                          'copy_X' : [True],
                                          'tol' : [0.0001, 0.001, 0.00001, 0.0005],
                                          'warm_start' : [True, False],
                                          'positive' : [True, False],
                                          'random_state' : [None],
                                          'selection' : ['Random', 'cyclic']
                                          }
                             },

                        'KernelRidge': \
                            {'model':KernelRidge(),
                             'name':'Kernel Ridge',
                             'param_grid' : {'alpha' : [.2, .5, 1],
                                          'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                                          'gamma' : [None],
                                          'degree' : [2, 3, 4],
                                          'coef0' : [1],
                                          'kernel_params' : [None]
                                          }
                             },

                        'SVR': \
                            {'model':SVR(),
                             'name':'SVR',
                             'param_grid' : {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                                          'degree' : [2, 3, 4],
                                          'gamma' : ['auto'],
                                          'coef0' : [0.0],
                                          'tol' : [0.001, 0.0001, 0.01],
                                          'C' : [0.7, 1.0, 1.2],
                                          'epsilon' : [0.1],
                                          'shrinking' : [True],
                                          'cache_size' : [200],
                                          'verbose' : [False],
                                          'max_iter' : [-1]
                                          }
                             },

                        'BaggingRegressor': \
                            {'model':BaggingRegressor(),
                             'name':'Bagging Regressor',
                             'param_grid' : {'base_estimator' : [None],
                                          'n_estimators' : [10, 20, 50, 100, 200, 250],
                                          'max_samples' : [1.0],
                                          'max_features' : [0.5, .08, 1.0],
                                          'bootstrap' : [True],
                                          'bootstrap_features' : [False],
                                          'oob_score' : [False],
                                          'warm_start' : [False],
                                          'n_jobs' : [-1],
                                          'random_state' : [None],
                                          'verbose' : [0]
                                          }
                             },

                        'ExtraTreesRegressor': \
                            {'model':ExtraTreesRegressor(),
                             'name':'Extra Trees Regressor',
                             'param_grid' : {'n_estimators' : [10, 20, 50, 100, 200, 250],
                                          'criterion' : ['mse', 'mae'],
                                          'max_depth' : [None, 2, 4, 6],
                                          'min_samples_split' : [2],
                                          'min_samples_leaf' : [1],
                                          'min_weight_fraction_leaf' : [0.0],
                                          'max_features' : [None, 'sqrt', 'log2'],
                                          'max_leaf_nodes' : [None],
                                          'min_impurity_decrease' : [0.0],
                                          'min_impurity_split' : [None],
                                          'bootstrap' : [False],
                                          'oob_score' : [False],
                                          'n_jobs' : [1],
                                          'random_state' : [None],
                                          'verbose' : [0],
                                          'warm_start' : [False]
                                          }
                             },

                        'KNeighborsRegressor': \
                            {'model':KNeighborsRegressor(),
                             'name':'K Neighbors Regressor',
                             'param_grid' : {'n_neighbors' : [2, 3, 5, 10, 15, 20, 25],
                                          'weights' : ['uniform', 'distance'],
                                          'algorithm' : ['kd_tree', 'ball_tree'],
                                          'leaf_size' : [2, 10, 30, 100],
                                          'p' : [1, 2],
                                          'metric' : ['minkowskis'],
                                          'metric_params' : [None],
                                          'n_jobs' : [-1]
                                          }
                             },

                        'RadiusNeighborsRegressor': \
                            {'model':RadiusNeighborsRegressor(),
                             'name':'Radius Neighbors Regressor',
                             'param_grid' : {'radius' : [1.0],
                                          'weights' : ['uniform', 'distance'],
                                          'algorithm' : ['kd_tree', 'ball_tree'],
                                          'leaf_size' : [2, 10, 30, 100],
                                          'p' : [1, 2],
                                          'metric' : ['minkowski'],
                                          'metric_params' : [None]
                                          }
                             },

                        'DecisionTreeRegressor': \
                            {'model':DecisionTreeRegressor(),
                             'name':'Decision Tree Regressor',
                             'param_grid' : {'criterion' : ['mse', 'mae'],
                                          'splitter' : ['best', 'random'],
                                          'max_depth' : [None, 2, 4, 6],
                                          'min_samples_split' : [2],
                                          'min_samples_leaf' : [1],
                                          'min_weight_fraction_leaf' : [0.0],
                                          'max_features' : [None, 'sqrt', 'log2'],
                                          'random_state' : [None],
                                          'max_leaf_nodes' : [None],
                                          'min_impurity_decrease' : [0.0],
                                          'min_impurity_split' : [None],
                                          'presort' : [False]
                                          }
                             },

                        'AdaBoostRegressor': \
                            {'model':AdaBoostRegressor(),
                             'name':'AdaBoost',
                             'param_grid' : {'base_estimator' : None,
                                          'n_estimators' : [20, 50, 100, 200],
                                          'learning_rate' : [1.0, 0.7, 1.3],
                                          'loss' : ['linear', 'square', 'exponential'],
                                          'random_state' : [None]
                                          }
                             },

                        'LinearRegression': \
                            {'model':LinearRegression(),
                             'name':'Linear Regression',
                             'param_grid' : {'fit_intercept' : [True],
                                          'normalize' : [False],
                                          'copy_X' : [True],
                                          'n_jobs' : [-1]
                                          }
                             },

                        'SGDRegressor': \
                            {'model':SGDRegressor(),
                            'name':'Stochastic Gradient Descent',
                            'param_grid' : {'loss' : ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                                         'penalty' : ['l2', 'l1'],
                                         'alpha' : [0.0001, 0.001, 0.00001],
                                         'l1_ratio' : [0.1, 0.15, 0.3, 0.5],
                                         'fit_intercept' : [True],
                                         'max_iter' : [None],
                                         'tol' : [None, 1e-4, 1e-2],
                                         'shuffle' : [True],
                                         'verbose' : [0],
                                         'epsilon' : [0.1],
                                         'random_state' : [None],
                                         'learning_rate' : ['invscaling', 'optimal'],
                                         'eta0' : [0.01],
                                         'power_t' : [.1, 0.25, .3],
                                         'warm_start' : [False],
                                         'average' : [False],
                                         'n_iter' : [None]
                                         }
                            },
                    }

        models = {model:model_options[model] for model in \
                    model_options.keys() if model in self.base_models.keys()}

        return models
