# Machine Learning
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import (KNeighborsRegressor, KNeighborsClassifier,
                               RadiusNeighborsRegressor, RadiusNeighborsClassifier)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              AdaBoostRegressor, AdaBoostClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              BaggingRegressor, BaggingClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.linear_model import (BayesianRidge,  RidgeClassifier,
                                  SGDRegressor, SGDClassifier,
                                  LinearRegression, LogisticRegression,
                                  Lasso, ElasticNet)


regression_options = {

                'MLPRegressor': {
                    'model':MLPRegressor(learning_rate = 'adaptive',
                                          max_iter=500,
                                          learning_rate_init=.005),
                    'name':'MLP NN'
                     },

                'RandomForestRegressor': {
                    'model':RandomForestRegressor(n_estimators = 20,
                                                   max_features = 2),
                    'name':'Random Forest'
                     },

                'BayesianRidge': {
                    'model':BayesianRidge(),
                    'name':'Bayesian Ridge'
                     },

                'Lasso': {
                    'model':Lasso(),
                    'name':'Lasso Regressor'
                     },

                'GradientBoostingRegressor': {
                    'model':GradientBoostingRegressor(max_features=2),
                    'name':'Gradient Boost'
                    },

                'ElasticNet': {
                    'model':ElasticNet(selection='random'),
                    'name':'Elastic Net Regressor'
                    },

                'KernelRidge': {
                    'model':KernelRidge(),
                    'name':'Kernel Ridge'
                    },

                'SVR': {
                    'model':SVR(),
                    'name':'SVR'
                    },

                'BaggingRegressor': {
                    'model':BaggingRegressor(),#base_estimator = LinearRegression()),
                    'name':'Bagging Regressor'
                    },

                'ExtraTreesRegressor': {
                    'model':ExtraTreesRegressor(),
                    'name':'Extra Trees Regressor'
                    },

                'KNeighborsRegressor': {
                    'model':KNeighborsRegressor(),
                    'name':'K Neighbors Regressor'
                    },

                'DecisionTreeRegressor': {
                    'model':DecisionTreeRegressor(),
                    'name':'Decision Tree Regressor'
                    },

                'AdaBoostRegressor': {
                    'model':AdaBoostRegressor(),#base_estimator = LinearRegression()),
                    'name':'AdaBoost'
                    },

                'LinearRegression': {
                    'model':LinearRegression(),
                    'name':'Linear Regression'
                    },

                'SGDRegressor': {
                    'model':SGDRegressor(max_iter=1000),
                    'name':'Stochastic Gradient Descent'
                    },
            }


classification_options = {

                'MLPClassifier': {
                    'model':MLPClassifier(learning_rate = 'adaptive',
                                          max_iter=500,
                                          learning_rate_init=.005),
                    'name':'MLP NN'
                     },

                'RandomForestClassifier': {
                    'model':RandomForestClassifier(n_estimators = 20,
                                                   max_features = 2),
                    'name':'Random Forest'
                     },

                'RidgeClassifier': {
                    'model':RidgeClassifier(),
                    'name':'Ridge Classifier'
                     },

                'GradientBoostingClassifier': {
                    'model':GradientBoostingClassifier(max_features=2),
                    'name':'Gradient Boost'
                    },

                'SVC': {
                    'model':SVC(),
                    'name':'SVC'
                    },

                'BaggingClassifier': {
                    'model':BaggingClassifier(),#base_estimator = LinearRegression()),
                    'name':'Bagging Classifier'
                    },

                'ExtraTreesClassifier': {
                    'model':ExtraTreesClassifier(),
                    'name':'Extra Trees Classifier'
                    },

                'KNeighborsClassifier': {
                    'model':KNeighborsClassifier(),
                    'name':'K Neighbors Classifier'
                    },

                'DecisionTreeClassifier': {
                    'model':DecisionTreeClassifier(),
                    'name':'Decision Tree Classifier'
                    },

                'AdaBoostClassifier': {
                    'model':AdaBoostClassifier(),#base_estimator = LinearRegression()),
                    'name':'AdaBoost'
                    },

                'LogisticRegression': {
                    'model':LogisticRegression(),
                    'name':'Logistic Regression'
                    },

                'SGDClassifier': {
                    'model':SGDClassifier(max_iter=1000),
                    'name':'Stochastic Gradient Descent'
                    },
            }
