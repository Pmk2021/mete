from dynaMETE import dynaMETE
from transition_function import transition_function
import numpy as np
import pytensor
import time

def f(n,states):
    dndt_f = n**2 + n
    dndt = dndt_f + 1/pytensor.tensor.dot(states,[1])
    return  0.0002 * n -  0.0002 * n**2 - 1/(10 + states[0])

f_func = transition_function(f, "F")

print("A")
mete = dynaMETE([1000], [f_func], derivatives = [0])

a = time.time()
for i in range(100):
    print(i)
    mete.update(0.01)
    print(mete.cur_state)
print(time.time() - a)