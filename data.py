"""
This module deals loading and transforming data
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from keras.utils import to_categorical


def load_data(filepath):
    """
    This function loads the train, validation and test data from a folder.
    It also normalizes the data along each histone signal (column) and returns the
    final data in numpy format.

    Parameters
    ----------
    filepath : `string`
        The path to the data files

    Returns
    -------
    `tuple`
        The normalized train, valid and test data
    """

    # Read all the data
    train = pd.read_csv(filepath+'/train.csv', header=None)
    test = pd.read_csv(filepath+'/test.csv', header=None)
    valid = pd.read_csv(filepath+'/valid.csv', header=None)


    # Extract the input and output
    y_train = train[7]
    x_train = train.drop([0, 1, 7], axis=1)

    y_test = test[7]
    x_test = test.drop([0, 1, 7], axis=1)

    y_valid = valid[7]
    x_valid = valid.drop([0, 1, 7], axis=1)


    # Convert the dataframes to numpy arrays
    x_train = np.array(x_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    x_valid = np.array(x_valid, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)
    y_valid = np.array(y_valid, dtype=float)

    # Reshape the arrays to get all the predictions for a particular gene together
    y_train = y_train.reshape((y_train.shape[0]//100, 100))
    y_train = y_train[:, 0]
    y_test = y_test.reshape((y_test.shape[0]//100, 100))
    y_test = y_test[:, 0]
    y_valid = y_valid.reshape((y_valid.shape[0]//100, 100))
    y_valid = y_valid[:, 0]

    # Similarly reshape the input variables
    x_train = x_train.reshape((x_train.shape[0]//100,100,x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0]//100,100,x_test.shape[1]))
    x_valid = x_valid.reshape((x_valid.shape[0]//100,100,x_valid.shape[1]))


    # Now for each train, valid and test normalize the data column wise and also add an
    # additional feature - the normalized sum of all modifications

    for i in range(x_train.shape[0]):
        s = x_train[i,:,:]
        s = normalize(s, axis=1, norm='max')
        x_train[i,:,:] = s

    a = x_train.sum(1)
    a = normalize(a, axis=1, norm='max')
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_train = np.c_[x_train, a]
    x_train = x_train.reshape((x_train.shape[0],-1,1))


    for i in range(x_test.shape[0]):
        s = x_test[i,:,:]
        s = normalize(s, axis=1, norm='max')
        x_test[i,:,:] = s

    a = x_test.sum(1)
    a = normalize(a, axis=1, norm='max')
    x_test = x_test.reshape((x_test.shape[0],-1))
    x_test = np.c_[x_test, a]
    x_test = x_test.reshape((x_test.shape[0],-1,1))


    for i in range(x_valid.shape[0]):
        s = x_valid[i,:,:]
        s = normalize(s, axis=1, norm='max')
        x_valid[i,:,:] = s

    a = x_valid.sum(1)
    a = normalize(a, axis=1, norm='max')
    x_valid = x_valid.reshape((x_valid.shape[0],-1))
    x_valid = np.c_[x_valid, a]
    x_valid = x_valid.reshape((x_valid.shape[0],-1,1))

    # Convert labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_valid = to_categorical(y_valid)


    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_data_mix(args):
    """
    This function also loads data for every cell but it also mixes the data for different cell types.
    It only mizes the train and validation data but the test is done individually on each cell

    Parameters
    ----------
    args : `dict`
        The argument dictionary

    Returns
    -------
    `tuple`
        The train, test and validation data
    """

    # Placeholders for the mixed data
    X = []
    Y = []

    # For each cell type
    cells = os.listdir(args.data_dir)
    
    for cell in cells:

        # The data path
        filepath = args.data_dir+'/'+cell+'/classification'
        train = pd.read_csv(filepath+'/train.csv', header=None)
        test = pd.read_csv(filepath+'/test.csv', header=None)
        valid = pd.read_csv(filepath+'/valid.csv', header=None)


        y_train = train[7]
        x_train = train.drop([0, 1, 7], axis=1)

        y_test = test[7]
        x_test = test.drop([0, 1, 7], axis=1)

        y_valid = valid[7]
        x_valid = valid.drop([0, 1, 7], axis=1)


        x_train = np.array(x_train, dtype=float)
        x_valid = np.array(x_valid, dtype=float)
        x_test = np.array(x_test, dtype=float)
        y_train = np.array(y_train, dtype=float)
        y_valid = np.array(y_valid, dtype=float)
        y_test = np.array(y_test, dtype=float)


        y_train = y_train.reshape((y_train.shape[0]//100, 100))
        y_train = y_train[:, 0]
        y_valid = y_valid.reshape((y_valid.shape[0]//100, 100))
        y_valid = y_valid[:, 0]
        y_test = y_test.reshape((y_test.shape[0]//100, 100))
        y_test = y_test[:, 0]


        x_train = x_train.reshape((x_train.shape[0]//100,100,x_train.shape[1]))
        x_valid = x_valid.reshape((x_valid.shape[0]//100,100,x_valid.shape[1]))
        x_test = x_valid.reshape((x_test.shape[0]//100,100,x_test.shape[1]))


        for i in range(x_train.shape[0]):
            s = x_train[i,:,:]
            s = normalize(s, axis=1, norm='max')
            x_train[i,:,:] = s

        a = x_train.sum(1)
        a = normalize(a, axis=1, norm='max')
        x_train = x_train.reshape((x_train.shape[0],-1))
        x_train = np.c_[x_train, a]
        x_train = x_train.reshape((x_train.shape[0],-1,1))


        for i in range(x_valid.shape[0]):
            s = x_valid[i,:,:]
            s = normalize(s, axis=1, norm='max')
            x_valid[i,:,:] = s

        a = x_valid.sum(1)
        a = normalize(a, axis=1, norm='max')
        x_valid = x_valid.reshape((x_valid.shape[0],-1))
        x_valid = np.c_[x_valid, a]
        x_valid = x_valid.reshape((x_valid.shape[0],-1,1))

        for i in range(x_test.shape[0]):
            s = x_test[i,:,:]
            s = normalize(s, axis=1, norm='max')
            x_test[i,:,:] = s

        a = x_test.sum(1)
        a = normalize(a, axis=1, norm='max')
        x_test = x_test.reshape((x_test.shape[0],-1))
        x_test = np.c_[x_test, a]
        x_test = x_test.reshape((x_test.shape[0],-1,1))


        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)
        y_test = to_categorical(y_test)

        X.append(x_train)
        X.append(x_valid)
        Y.append(y_train)
        Y.append(y_valid)

    X = np.array(X)
    Y = np.array(Y)

    X = X.reshape((-1,505,1))
    Y = Y.reshape((-1,2))

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.3, shuffle = True, random_state =2)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
