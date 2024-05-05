import numpy as np
from scipy.stats import norm


class DiffusionProcess:

    def __init__ (self, r, sigma):
        self.r = r
        self.sigma = sigma

    def gbm (self, S0, T, N):
        w = norm.rvs(0, 1, N)
        S = S0 * np.exp((self.r - 0.5 * self.sigma ** 2) * T + self.sigma * np.sqrt(T) * w)
        return S
