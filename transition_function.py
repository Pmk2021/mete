import numpy as np
import pytensor


class transition_function:

    def __init__(self, function, name):
        """
        :param constraint: function that takes in state and returns symbolic expression of constraint function
        :param function: symbolic expression of transition function
        """
        self.function = function
        self.name = name

    def evaluate(self, states, n: float):

        return self.function(n, states)

    def get_function(self):
        """
        :return: pytensor function that takes in a vector of integers and evaluates self.function on every element
        """
        individuals = pytensor.tensor.vector()
        return pytensor.function([individuals], self.function(individuals))
