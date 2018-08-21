from pmlb import fetch_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from ensemble import Ensemble


def get_benchmark_data(dataset_name, split_percent=.9):
    X,y = fetch_data(dataset_name, return_X_y=True)
    split = int(len(X)*.9)
    return X[:split,:], y[:split], X[split:,:], y[split:]


def regression_benchmark():

    datasets = [
        '529_pollen',
        '622_fri_c2_1000_50',
        '649_fri_c0_500_5',
        '690_visualizing_galaxy',
        '1193_BNG_lowbwt'
        ]

    models = {
        'Linear Regression: ':LinearRegression(),
        'Random Forest Regressor: ':RandomForestRegressor(),
        'MLP Neural Network: ':MLPRegressor(max_iter=500),
        'Support Vector Regressor: ':SVR()
        }

    print('\n', 'Model R2 Scores (1 is best):')

    for i, dataset in enumerate(datasets):

        Xtrain, ytrain, Xtest, ytest = get_benchmark_data(dataset)

        print('\n', f'Benchmark {i+1}:')
        print('--------------------------------')
        print(f'Dataset: {dataset}', '\n')

        for model_name, model in models.items():
            try:
                model.fit(Xtrain, ytrain)
                print(model_name, round(model.score(Xtest, ytest),4))
            except:
                print(f'Could not train {model.__name__}')


def classification_benchmark():

    datasets = [
        'breast-cancer',
        'cleveland-nominal',
        'horse-colic',
        'solar-flare_1'
        ]

    models = {
        'Logistic Regression: ':LogisticRegression(),
        'Random Forest Classifier: ':RandomForestClassifier(),
        'MLP Neural Network: ':MLPClassifier(max_iter=500),
        'Support Vector Classifier: ':SVC()
        }

    print('\n', 'Model Accuracy Scores (1 is best):')

    for i, dataset in enumerate(datasets):

        Xtrain, ytrain, Xtest, ytest = get_benchmark_data(dataset)

        print('\n', f'Benchmark {i+1}:')
        print('--------------------------------')
        print(f'Dataset: {dataset}', '\n')

        for model_name, model in models.items():
            model.fit(Xtrain, ytrain)
            print(model_name, round(model.score(Xtest, ytest),4))
