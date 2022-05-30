from src.fitness.base_ff_classes.base_ff import base_ff
import time
import shutil
import os


class my_ff(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        dir = "C:/Users/luigi/Documents/GitHub/PonyGE2/datasets/Glucose"

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        print("\n" + p)
        fitness = 0
        for trial in range(50):
            self.test_list = get_sample()
            m = max(self.test_list)
            d = {'test_list': self.test_list}
            try:
                t0 = time.time()
                exec(p, d)
                t1 = time.time()
                guess = d['return_val']
                fitness += len(p)
                v = abs(m - guess)
                if v <= 10 ** 6:
                    fitness += v
                else:
                    fitness = self.default_fitness
                    break
                if t1 - t0 < 10:
                    fitness = self.default_fitness
                    break
                else:
                    fitness += (t1 - t0) * 1000
            except:
                fitness = self.default_fitness
                break
        return fitness


def get_sample():
    # questa funzione apre il file target, prende un campione e lo chiude
    return


def copy_dir(src):
    # This function creates a copy of the src directory in the same folder
    delimiter = '/'
    split = src.split('/')
    name = split.pop()
    dst = delimiter.join(split) + "/" + name + "_copy"
    shutil.copytree(src, dst)
    return dst


def delete_dir(src):
    # This function deletes the target directory
    shutil.rmtree(src)
    return


def get_directories(src):
    # This function calculates the number of trials needed for the calculation
    directories = []
    for subdir, dirs, files in os.walk(src):
        directories.append(subdir)
    return directories


if __name__ == '__main__':
    print("Tests for fitness function module")

    data_dir = "C:/Users/luigi/Documents/GitHub/PonyGE2/datasets/Glucose"
    sample = "ws-training"

    copied_dir = copy_dir(data_dir)
    subdirs = get_directories(copied_dir)
    print(subdirs)
    trials = len(subdirs)

    for dir in subdirs:
        print("gg")
        #qua si deve:
        #leggere il csv
        #prendere una riga per volta
        #trasformarla in qualche modo
        #passarla alla funzione eval
        #vedere se si trova il guess
        #etcc...


    delete_dir(copied_dir)
