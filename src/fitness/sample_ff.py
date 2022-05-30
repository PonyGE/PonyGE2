from fitness.base_ff_classes.base_ff import base_ff
import random
import time


class sample_ff(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        print("\n" + p)
        fitness = 0
        for trial in range(50):
            self.test_list = generate_list()
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

def generate_list():
    return [random.randint(0, round(random.random() * 90 + 10, 0)) for i in range(9)]
