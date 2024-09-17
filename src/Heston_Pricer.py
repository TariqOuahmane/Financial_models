import numpy as np
from COS_method import COSMethod


class Heston_pricer:
    """
    price the option with hetson model be mean of :
        -Fourier inversion
        -Monte Carlo
    """

    def __init__ (self, option_info, process_info):
        """
        option_info : it contains S, K and T
        process_info : of type heston it contains the interest rate r , sigma,theta , kappa and rho
        """
        # heston parameter
        self.r = process_info.mu
        self.sigma = process_info.sigma
        self.theta = process_info.theta
        self.kappa = process_info.kappa
        self.rho = process_info.rho

        # option parameters
        self.S0 = option_info.S0
        self.v0 = option_info.v0
        self.K = option_info.K
        self.T = option_info.T

        self.exercise = option_info.exercise
        self.payoff_value = option_info.payoff_value  # value of the payof (S-K)+ or (K-s)+
        self.payoff_type = option_info.payoff_type  # type of the payoff "european" or "american"
        self.ch_function = process_info.ch_function

        self.cos_obj = COSMethod(option_info=option_info, process_info=process_info)

    #the cos method for the option price
    def Option_price_COS_method(self, N: int, L: int):
        return self.cos_obj.COS_Price(N=N, L=L, ch_function=self.ch_function)
