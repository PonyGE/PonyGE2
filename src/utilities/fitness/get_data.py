from os import path

import numpy as np
import pandas as pd
from algorithm.parameters import params


def get_Xy_train_test_separate(train_filename, test_filename, skip_header=0):
    """
    Read in training and testing data files, and split each into X
    (all columns up to last) and y (last column). The data files should
    contain one row per training example.
    
    :param train_filename: The file name of the training dataset.
    :param test_filename: The file name of the testing dataset.
    :param skip_header: The number of header lines to skip.
    :return: Parsed Pandas DataFrames of training and testing input (x) and
    output (y) data.
    """

    # Read in all training data.
    train_Xy = pd.read_csv(train_filename, sep=None)
    
    try:
        # Separate out input (X) and output (y) data.
        train_X = train_Xy.iloc[:, :-1] # all columns but last
        train_y = train_Xy.iloc[:, -1]  # last column

    except IndexError:
        s = "utilities.fitness.get_data.get_Xy_train_test_separate\n" \
            "Error: specified delimiter '%s' incorrectly parses training " \
            "data." % delimiter
        raise Exception(s)

    if test_filename:
        # Read in all testing data.
        test_Xy = pd.read_csv(test_filename, sep=None)

        # Separate out input (X) and output (y) data.
        test_X = test_Xy.iloc[:, :-1] # all columns but last
        test_y = test_Xy.iloc[:, -1]  # last column

    else:
        test_X, test_y = None, None

    return train_X, train_y, test_X, test_y


def get_data(train, test):
    """
    Return the training and test data for the current experiment.
    
    :param train: The desired training dataset.
    :param test: The desired testing dataset.
    :return: The parsed data contained in the dataset files.
    """

    # Get the path to the training dataset.
    train_set = path.join("..", "datasets", train)

    if test:
        # Get the path to the testing dataset.
        test_set = path.join("..", "datasets", test)

    else:
        # There is no testing dataset used.
        test_set = None

    # Read in the training and testing datasets from the specified files.
    training_in, training_out, test_in, \
    test_out = get_Xy_train_test_separate(train_set, test_set, skip_header=1)

    return training_in, training_out, test_in, test_out
