from fitness.base_ff_classes.base_ff import base_ff
import time
import shutil
import os
import random
import pandas as pd
import pickle
import math
from sklearn.metrics import mean_squared_error

generation_range = 9
data_dir = "datasets/Glucose"
training_ext = "ws-training"
testing_ext = "ws-testing"
train_file = "datasets\\Glucose\\540\\540-ws-training.csv"
train_absolute = "C:\\Users\\luigi\\Documents\\GitHub\\PonyGE2\\datasets\\Glucose\\540\\540-ws-training.csv"


def variables_substitution(p, tuple):
    var = 'X'
    while True:
        index = p.find(var)
        if index == -1:
            break
        s = p[index:index + 2]
        i = int(s[1:]) - 1
        sub = str(tuple[i])
        p = p.replace(s, sub)
    return p


class my_ff(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        #print("\n" + p)
        fitness = 0
        file = pd.read_csv(train_absolute, skiprows=1, header=None)
        for x in range(file.shape[0]):
            tuple = file.loc[x].tolist()
            function = variables_substitution(p, tuple)
            self.test_list = tuple
            try:
                t0 = time.time()
                guess = eval(function)
                t1 = time.time()
                fitness += mean_squared_error(guess, tuple[54], squared=False)
            except:
                fitness = self.default_fitness
                break
        return fitness


def open_train_file(directory, extension):
    # This function, using the extension, finds the train file, opens it using pandas e returns the dataframe
    file = None
    print(directory + "root directory")
    for root, dirs, files in os.walk(directory):
        print(root, dirs, files)
        for filename in files:
            if filename.find(extension) > 0:
                file = directory + "/" + filename
                print("dai tocca qui ")
                print(file)
    if file is not None:
        print("SONO QUI")
        df = pd.read_csv(file, skiprows=1, header=None)
        print(df)
        return df
    else:
        return None


def get_directories(src):
    # This function calculates the number of trials needed for the calculation
    directories = []
    for subdir, dirs, files in os.walk(src):
        directories.append(subdir)
    return directories


def get_normalized_lists(file):
    glucose = []
    insuline = []
    carbos = []
    for x in range(file.shape[0]):
        line = file.loc[x].tolist()
        variables = int(len(line) / 3)
        # print(line, line.__class__)
        g = []
        i = []
        c = []

        for k in range(variables):
            g.append(line[k * 3])
            i.append(line[k * 3 + 1])
            c.append(line[k * 3 + 2])

        glucose.append(g)
        insuline.append(i)
        carbos.append(c)
    return glucose, insuline, carbos


def mane():
    print("Tests for fitness function module")

    # copied_dir = copy_dir(data_dir)
    subdirs = get_directories(data_dir)
    subdirs.pop(0)
    print(subdirs)
    trials = len(subdirs)

    glucose = []
    insuline = []
    carbos = []

    for dir in subdirs:
        file = open_train_file(dir, extension=training_ext)
        glucose, insuline, carbos = get_normalized_lists(file)

        # qua si deve:
        # leggere il csv
        # prendere una riga per volta
        # trasformarla in qualche modo
        # passarla alla funzione eval
        # vedere se si trova il guess
        # etcc...

    # delete_dir(copied_dir)


if __name__ == '__main__':
    string = "X2 X3 X2 X3"
    var = '6'
    tuple = [0.1212313, 1.124155515, 2.2352363422]
    variables_substitution(string, tuple)

    x = eval("(10 + 32 - 36) + 94*math.pow(10, -1) -89*math.pow(10, -4)")
    print(x)

