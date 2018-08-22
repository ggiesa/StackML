



class optimize:
    def __init__(self, ensemble_object):

        # Get attributes from ensemble
        self.ensemble =         ensemble_object
        self.base_models =      self.ensemble.base_models
        self.combiner =         self.ensemble.combiner
        self.Xtrain =           self.ensemble.Xtrain
        self.ytrain =           self.ensemble.Ytrain
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
            results = gsearch.fit(self.Xtrain, self.ytrain)

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
