import numpy as np
import pytensor
import transition_function
from scipy.optimize import minimize
import time

N_INDEX = 0

class dynaMETE:
    def __init__(self, states: list, constraints: list[transition_function], lambdas: list[float] = None, derivatives: list = None) -> None:
        self.cur_state = states
        self.lambdas = lambdas
        self.past_states = list()
        self.past_lambdas = list()
        self.derivatives = derivatives
        self.constraints = constraints
        self.state_labels = dict()
        if derivatives is None:
            derivatives = np.zeros(len(constraints))

        ##Calculate necessary functions ahead of time
        self.min_grad, self.min_func, self.deriv_func = self.get_deriv_func()

        ##Calculate functiont to get derivatives

        if lambdas is None:
            self.lambdas = np.ones(len(self.constraints) + 1)
            lambdas = self.get_new_lambdas()



    def get_deriv_func(self):
        self.last_update = self.cur_state[N_INDEX]
        lambdas = pytensor.tensor.vector('lambdas')
        derivs = pytensor.tensor.vector('derivatives')
        states = pytensor.tensor.vector('states')


        diff1 = self.get_derivatives(lambdas, states) - derivs
        diff2 = self.get_N(lambdas) - pytensor.tensor.sum(states * np.array([1] + [0] * (len(self.cur_state) - 1)))

        diff = pytensor.tensor.sum(np.dot(diff1,diff1)) + pytensor.tensor.sum(np.dot(diff2,diff2))

        
        min_func = pytensor.function([lambdas, derivs, states], diff1, on_unused_input='ignore')

        gs = pytensor.tensor.grad(diff, lambdas)

        grad = pytensor.function([lambdas, derivs, states], gs,  on_unused_input='ignore')
        
        deriv_func = pytensor.function([lambdas,states], self.get_derivatives(lambdas, states),  on_unused_input='ignore')

        return grad, min_func, deriv_func

    def update(self, time_step) -> None:
        """
        Get update step
        :return: None
        """
        #To save on time, all the functions are precompiled and optimized. They're only changed once N changes by a decent amount since
        #there isn't a large difference between using vs not using the last few indices

        if  (self.cur_state[N_INDEX] - self.last_update)**2 > 10:
            self.min_grad, self.min_func, self.deriv_func = self.get_deriv_func()
        

        #Add current state and lambdas to history
        self.past_states.append(list(self.cur_state))
        self.past_lambdas.append(list(self.lambdas))


        self.cur_state = [self.cur_state[i] + self.derivatives[i] * time_step for i in range(len(self.cur_state))] #update previous state
        
        self.lambdas = self.get_new_lambdas()

        new_derivatives = self.deriv_func(self.lambdas, self.cur_state)

 
        for i in range(len(self.derivatives)):
            self.derivatives[i] = new_derivatives[0]
            new_derivatives = new_derivatives[1:]
        
    

    def get_new_lambdas(self):
        
        def min_func(l):
            return self.min_func(l, self.cur_state, self.derivatives)
        
        def grad(l):
            return self.min_grad(l, self.cur_state, self.derivatives)
    
        def deriv(l):
            return self.deriv_func(l, self.cur_state)
        
        
        a = time.time()
        a,b,c = self.get_deriv_func()


        best_lambdas = minimize(min_func, self.lambdas, jac=grad, method='Newton-CG',  options={'xtol': 1e-8, 'disp': False})

        return best_lambdas.x

    def get_derivatives(self, lambdas, states=None):
        '''
        Calculates derivatives for every state variable using transition variables

        :return: list or pytensor function with derivatives
        '''
        if(states==None):
            states=self.cur_state

        dist = self.get_r(lambdas)

        derivatives = list()
        n = np.array([i for i in range(int(self.cur_state[N_INDEX]))])

        for i in range(len(self.constraints)):
            derivatives.append(pytensor.tensor.dot(dist, self.constraints[i].evaluate(states, n)))
  
        return derivatives

    def get_r(self, lambdas):
        '''
        Calculates distribution using current lambdas. Can either return an exact list, or a pytensor function depending on 
        whether lambdas is a list or pytensor variable
        
        :return: list or pytensor function with R calculated on every value from 0 to N
        '''

 
        dist = np.ones(int(self.cur_state[N_INDEX]))
        n = np.array([i for i in range(1, int(self.cur_state[N_INDEX]) + 1)])

 
        for i in range(len(self.constraints)):
            dist *= np.e**(-lambdas[1+i] * self.constraints[i].evaluate(self.cur_state,n))
        
        dist *= np.e**(-lambdas[0] * n)
        

        return dist/pytensor.tensor.sum(dist)


    def get_N(self, lambdas):
        '''
        Calculates N
        '''
        dist = self.get_r(lambdas)
        d = list()
        n = np.array([i for i in range(int(self.cur_state[N_INDEX]))])

        return pytensor.tensor.dot(dist, n)

