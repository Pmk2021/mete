from dynaMETE import dynaMETE
from transition_function import transition_function
import numpy as np
np.bool = np.bool_
import theano


def f(states):
    n =  theano.tensor.vector('n')
    dndt_f = n**2 + n
    dndt = dndt_f
    return  theano.function([n], dndt)

f_func = transition_function(f, "F")

mete = dynaMETE({"N":1000}, [f_func], np.array([0.0]), derivatives ={"N":0})

mete.update(0.1)