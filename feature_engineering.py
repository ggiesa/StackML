import pandas as pd


class FeatureEngineering:
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
            self.ytrain = self.Y[:self.split_ind]
            self.ytest = self.Y[self.split_ind:]
        else:
            self.ytrain = self.Y.iloc[:self.split_ind, :]
            self.ytest = self.Y.iloc[self.split_ind:, :]

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
            return self.Xtrain, self.ytrain, self.Xtest, self.ytest
        else:
            if self.y_gen == 0:
                return self.Xtrain, self.ytrain, self.Xtest, self.ytest
            else:
                return self.Xtrain, self.ytrain.iloc[:, self.y_gen], \
                       self.Xtest,  self.ytest.iloc[:,  self.y_gen]
