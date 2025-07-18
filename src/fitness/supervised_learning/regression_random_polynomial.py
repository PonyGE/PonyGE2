import itertools
import random

import numpy as np
from algorithm.parameters import params
from fitness.supervised_learning.regression import regression
from utilities.fitness.error_metric import rmse


class regression_random_polynomial(regression):
    """Fitness function for regression of random polynomials. We
    specialise the regression fitness function.

    The user must pass in the degree, n_vars, and n_samples. A random
    polynomial of the given degree and number of variables is
    generated (ie random coefficients are generated). It is evaluated
    at n_samples, randomly generated. That gives a target dataset for
    training. This is accomplished with, eg, --extra_parameters 5 1
    20, giving degree 5, 1 variable, 20 fitness cases.

    If the user requires test on unseen data, the polynomial is
    evaluated at a new sample of points, the same size. That is
    accomplished with --dataset_test Dummy (the "Dummy" argument would
    usually be a filename, but we do not attempt to read a file, but
    just take it as a switch to generate the separate test set).

    An example command-line is then:

    python ponyge.py --generations 10 --population 10 --fitness
    supervised_learning.regression_random_polynomial
    --extra_parameters 5 1 20 --grammar
    supervised_learning/supervised_learning.bnf --dataset_test Dummy

    """

    def __init__(self):
        # Don't call super().__init__() because it reads the training
        # (and test) data from files. We'll do everything else it
        # would have done.

        degree, n_vars, n_samples = map(int, params['EXTRA_PARAMETERS'])

        # may be needed if the grammar uses GE_RANGE:dataset_n_vars
        self.n_vars = n_vars

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = rmse

        self.maximise = params['ERROR_METRIC'].maximise

        # create a random polynomial
        p = Polynomial.from_random(degree, n_vars)

        # generate a set of fitness cases for training
        self.training_in = np.random.random((n_vars, n_samples))
        self.training_exp = p.eval(self.training_in)

        # if we want a separate test set, generate a set of fitness
        # cases for it
        if params['DATASET_TEST']:
            self.test_in = np.random.random((n_vars, n_samples))
            self.test_exp = p.eval(self.training_in)


class Polynomial:
    """A polynomial of a given degree and a given number of variables,
    with one coefficient for each term."""

    def __init__(self, degree, n_vars, coefs):
        """Constructor for the case where we already have coefficients."""
        self.degree = degree
        self.n_vars = n_vars
        self.coefs = coefs

    @classmethod
    def from_random(cls, degree, n_vars):
        """Constructor for the case where we want random coefficients."""
        coefs = [2 * (random.random() - 0.5) for i in
                 Polynomial.terms(degree, n_vars)]
        p = Polynomial(degree, n_vars, coefs)
        return p

    @staticmethod
    def terms(degree, n_vars):
        """Generator for the possible terms of a polynomial
        given the degree and number of variables."""
        for pows in itertools.product(range(degree + 1), repeat=n_vars):
            if sum(pows) <= degree:
                yield pows

    def eval(self, x):
        """Evaluate the polynomial at a set of points x."""
        assert x.shape[0] == self.n_vars
        result = np.zeros(x.shape[1])  # same length as a column of x
        for coef, pows in zip(self.coefs,
                              self.terms(self.degree, self.n_vars)):
            tmp = np.ones(x.shape[1])
            for (xi, pow) in zip(x, pows):
                tmp *= (xi ** pow)
            tmp *= coef
            result += tmp
        return result

    def __str__(self):
        """Pretty-print a polynomial, rounding the coefficients."""

        def s(pows):
            if sum(pows):
                return "*" + "*".join("x[%d]**%d" % (i, powi)
                                      for i, powi in enumerate(pows) if
                                      powi > 0)
            else:
                # this term is a const so the coef on its own is enough
                return ""

        return " + ".join("%.3f%s" % (coef, s(pows))
                          for (coef, pows) in
                          zip(self.coefs,
                              Polynomial.terms(self.degree, self.n_vars)))
