import numpy as np
np.bool = np.bool_
import theano
import transition_function
from scipy.optimize import minimize

class dynaMETE:
    def __init__(self, states: dict, constraints: list[transition_function], lambdas: list[float] = None, derivatives: dict = None) -> None:
        self.labels = list(states.keys())
        self.cur_state = states
        self.lambdas = lambdas
        self.past_states = list()
        self.past_lambdas = list()
        self.derivatives = derivatives
        self.constraints = constraints
        self.state_labels = dict()
        if derivatives is None:
            derivatives = np.zeros(len(constraints))
        if lambdas is None:
            lambdas = np.zeros(len(constraints))

    def update(self, time_step) -> None:
        """
        Get update step
        :return: None
        """
        #Add current state and lambdas to history
        self.past_states.append(list(self.cur_state))
        self.past_lambdas.append(list(self.lambdas))

        self.cur_state = {i:self.cur_state[i] + self.derivatives[i] * time_step for i in self.cur_state} #update previous state

        R = self.get_r(self.lambdas)

        self.lambdas = self.get_new_lambdas(R)


        self.derivatives = self.get_derivatives()
    def get_new_lambdas(self, R):
        norm = 1/sum(R)

        lambdas = theano.tensor.vector('lambdas')

        print("DDD")
        print(self.get_derivatives([0], 1))

        diff = self.get_derivatives(lambdas, norm) - theano.tensor._shared(np.array([self.derivatives[i] for i in self.derivatives]))


        f1 = theano.function([lambdas], self.get_derivatives(lambdas, norm))

        diff = theano.function([lambdas],theano.tensor.dot(diff,diff))

        gs = theano.tensor.grad(diff, lambdas)

        print("AAA")
        best_lambdas = minimize(theano.tensor.sum(theano.dot(diff,diff),lambdas), self.lambdas, method='Newton-CG',
                       jac=gs,
                       options={'xtol': 1e-8, 'disp': True})

        return best_lambdas

    def get_derivatives(self, lambdas, norm):
        dist = self.get_r(lambdas)

        derivatives = list()
        n = np.array([i for i in range(int(self.cur_state["N"]))])

        for i in range(len(self.constraints)):
            print(len(dist))
            print(len(self.constraints[i].evaluate(self.cur_state,n)))
            derivatives.append(theano.dot(dist, self.constraints[i].evaluate(self.cur_state,n)))

        return derivatives

    def get_r(self, lambdas):

        dist = np.zeros(int(self.cur_state["N"]))
        n = np.array([i for i in range(int(self.cur_state["N"]))])

        for i in range(len(self.lambdas)):
            dist *= np.e**(-lambdas[i] * self.constraints[i].evaluate(self.cur_state,n))

        return dist


    def get_N(self, lambdas):
        dist = self.get_r(lambdas)
        d = list()
        n = np.array([i for i in range(int(self.cur_state["N"]))])

        return theano.dot(dist, n)

