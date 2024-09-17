import numpy as np


class Option_Info:
    """
    class to define the params of the option S0,K,r,T, and v0 optional (spot variance)
    """

    def __init__ (self, S0, K, T, v0, payoff="call", exercise="European"):
        self.S0 = S0
        self.K = K
        self.T = T
        self.v0 = v0

        if self.S0 < 0 or self.K < 0 or self.T <= 0  or self.v0 < 0:
            raise ValueError('Error: Negative inputs not allowed')

        if exercise == "European" or exercise == "American":
            self.exercise = exercise
        else:
            raise ValueError("invalid exercise type")
        if payoff == "call" or payoff == "put":
            self.payoff_type = payoff
        else:
            raise ValueError("invalid payoff type .")

    def payoff_value(self, S):
        if self.payoff_type == 'call':
            payoff = np.maximum(S - self.K, 0)
        elif self.payoff_type == 'put':
            payoff = np.maximum(self.K - S, 0)
        else:
            raise ValueError("specify a 'put' or 'call' ")
        return payoff
