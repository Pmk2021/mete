import numpy as np
np.bool = np.bool_
import theano


class transition_function:

    def __init__(self, function, name):
        """
        :param constraint: function that takes in state and returns symbolic expression of constraint function
        :param function: symbolic expression of transition function
        """
        self.function = function
        self.name = name

    def evaluate(self, states: np.ndarray, n: float):
        return self.function(states)(n)

    def get_function(self):
        """
        :return: theano function that takes in a vector of integers and evaluates self.function on every element
        """
        individuals = theano.tensor.vector()
        return theano.function([individuals], self.function(individuals))
