import numpy as np
from scipy.stats import norm

j = complex(0.0, 1.0)


class BS_pricer:

    def __init__ (self, option_info, process_info):
        """
        :param option_info: it contains (S,K,T)
        :param process_info: type of the diffusion process it contain (r,mu,sigma)
        """
        self.r = process_info.mu
        self.sigma = process_info.sigma
        self.S0 = option_info.S0
        self.K = option_info.K
        self.T = option_info.T
        self.geometric_process = process_info.geometric_process
        self.payoff = option_info.payoff
        self.exercise = option_info.exercise
        self.ch_function=process_info.ch_f

    def BlackSholesPrice (self):
        """
        Calculates price for call option according to the formula.
        Formula: S*N(d1) - PresentValue(K)*N(d2)
        """
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be
        # exercised)

        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of
        # the option)
        d2 = (np.log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.payoff == 'call':
            return self.S0 * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)
        elif self.payoff == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def payoff_op (self, S):
        if self.payoff == 'call':
            payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put':
            payoff = np.maximum(self.K - S, 0)
        else:
            raise ValueError("specify a 'put' or 'call' ")
        return payoff

    def MonteCarlo (self, N: int, steps: int):
        S = self.geometric_process(S0=self.S0, N=N, T=self.T, num_steps=steps)
        payoff = self.payoff_op(S [-1])
        V = np.sum(payoff) / N
        return np.exp(-self.r * self.T) * V

    def COS_method (self, N: int, L: int):
        a = -L * np.sqrt(self.T)
        b = L * np.sqrt(self.T)
        k = np.linspace(0, N - 1, N)
        u = k * np.pi / (b - a)
        x0 = np.log(self.S0 / self.K)
        # the coeff for the option price
        H_k = self.CallPutCoefficients(a, b, k)
        temp = self.ch_function(u,self.T) * H_k
        temp [0] = temp [0] * 0.5
        mat = np.exp(j * np.dot((x0 - a), u))
        OptionVal = np.exp(-self.r * self.T) * np.real(temp.dot(mat)) * self.K
        return OptionVal

    # the coefficients H_k of the CS method
    def CallPutCoefficients (self, a, b, k):
        if self.payoff == "call":
            c = 0.0
            d = b
            if a < b and b < 0.0:
                H_k = np.zeros(len(k))
            else:
                H_k = 2.0 / (b - a) * (self.chi(a, b, c, d, k) - self.psi(a, b, c, d, k))

        if self.payoff == "put":
            c = a
            d = 0.0
            H_k = 2.0 / (b - a) * (-self.chi(a, b, c, d) + self.psi(a, b, c, d))

        return H_k

    # the characteristic function og GBM

    @staticmethod
    def chi (a, b, c, d, k):
        u = k * np.pi / (b - a)
        chi = 1.0 / (1 + u ** 2)
        expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                           * (c - a) / (b - a)) * np.exp(c)
        expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                             (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                               * np.pi * (c - a) / (
                                                                                                       b - a)) * np.exp(
            c)
        chi = chi * (expr1 + expr2)
        return chi

    @staticmethod
    def psi (a, b, c, d, k):
        psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
        psi [1:] = psi [1:] * (b - a) / (np.pi * k [1:])
        psi [0] = d - c
        return psi
