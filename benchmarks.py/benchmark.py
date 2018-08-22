from pmlb import fetch_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from stackml import StackML

from importlib import reload
import stackml as s
s = reload(s)




def get_benchmark_data(dataset_name, split_percent=.9):
    data = fetch_data(dataset_name)
    y = data.target
    X = data.drop('target', axis=1)
    split = int(len(X)*.9)
    return X.iloc[:split,:], y.iloc[:split], X.iloc[split:,:], y.iloc[split:]


def classification_benchmark(StackML_model=None, datasets = None, models=None):

    if not datasets:
        datasets = [
            'breast-cancer',
            'cleveland-nominal',
            'horse-colic',
            'solar-flare_1'
            ]

    if not models:
        models = {
            'Logistic Regression: ':LogisticRegression(),
            'Random Forest Classifier: ':RandomForestClassifier(),
            'MLP Neural Network: ':MLPClassifier(max_iter=500),
            'Support Vector Classifier: ':SVC()
            }

    if StackML_model:
        models['StackML: '] = StackML_model

    errors = []
    print('\n', 'Model Accuracy Scores (1 is best):')

    for i, dataset in enumerate(datasets):

        Xtrain, ytrain, Xtest, ytest = get_benchmark_data(dataset)

        print('\n', f'Benchmark {i+1}:')
        print('--------------------------------')
        print(f'Dataset: {dataset}', '\n')

        for model_name, model in models.items():

            try:
                model.fit(Xtrain, ytrain)
                print(model_name, round(model.score(Xtest, ytest),4))

            except Exception as err:
                print(f'Failed to train {model_name}')
                errors.append(err)

    if errors:
        print('\n', 'Errors:')
        for error in errors:
            print(error)



def regression_benchmark(StackML_model=None, datasets = None, models=None):

    if not datasets:
        datasets = [
            '529_pollen',
            '622_fri_c2_1000_50',
            '649_fri_c0_500_5',
            '690_visualizing_galaxy',
            ]

    if not models:
        models = {
            'Linear Regression: ':LinearRegression(),
            'Random Forest Regressor: ':RandomForestRegressor(),
            'MLP Neural Network: ':MLPRegressor(max_iter=500),
            'Support Vector Regressor: ':SVR()
            }


    if StackML_model:
        models['StackML: '] = StackML_model

    errors = []
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

            except Exception as err:
                print(f'Failed to train {model_name}')
                errors.append(err)

    if errors:
        print('\n', 'Errors:')
        for error in errors:
            print(error)



def test_all_regression(model):
    pass

def test_all_classification(model):
    pass


def quick_test():
    model = s.StackML(verbose=True)
    datasets = [
        '529_pollen',
        '622_fri_c2_1000_50',
        # '690_visualizing_galaxy',
        '649_fri_c0_500_5',
        ]

    for i, dataset in enumerate(datasets):
        Xtrain, ytrain, Xtest, ytest = get_benchmark_data(dataset)
        model.fit(Xtrain, ytrain)
        print(f'{dataset}:', round(model.score(Xtest, ytest),4))
        model.plot_prediction()



quick_test()
