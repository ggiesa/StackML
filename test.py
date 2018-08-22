from pmlb import fetch_data
from importlib import reload
from unittest import TestCase

import stackml as s
s = reload(s)



def regression_test_datasets():

    datasets = [
        'breast-cancer',
        'cleveland-nominal',
        'horse-colic',
        'solar-flare_1'
        ]

    for dataset in datasets:

        data = fetch_data(dataset)
        y = data.target
        X = data.drop('target', axis=1)
        split = int(len(X)*.9)
        Xtrain = X.iloc[:split,:]
        ytrain = y.iloc[:split]
        Xtest = X.iloc[split:,:]
        ytest = y.iloc[split:]

        yield (Xtrain, ytrain, Xtest, ytest)


def classification_test_datasets():

    datasets = [
        '529_pollen',
        '622_fri_c2_1000_50',
        '649_fri_c0_500_5',
        '690_visualizing_galaxy',
        ]

    for dataset in datasets:

        data = fetch_data(dataset)
        y = data.target
        X = data.drop('target', axis=1)
        split = int(len(X)*.9)
        Xtrain = X.iloc[:split,:]
        ytrain = y.iloc[:split]
        Xtest = X.iloc[split:,:]
        ytest = y.iloc[split:]

        yield (Xtrain, ytrain, Xtest, ytest)




def TestStackML:

    def test_cv_folds(self):
        pass

    def test_reuse_features(self):
        pass

    def test_shuffle(self):
        pass

    def test_timeseries(self):
        pass

    def test_base_models(self):
        pass

    def test_combiner(self):
        pass

    def test_plotting(self):
        pass

    def test_discard_models(self):
        pass

    def test_use_best_model(self):
        pass



class TestRegression(TestCase):
    pass


class TestClassification(TestCase):
    pass



if __name__ == '__main__':
    unittest.main()
