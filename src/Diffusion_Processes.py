import numpy as np
from scipy.stats import norm

j = complex(0.0, 1.0)


class GeometricBM:
    """
    class to define the diffusion process
    """

    def __init__ (self, r: float, sigma: float, mu: float):
        self.r = r
        self.mu = mu
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        else:
            self.sigma = sigma

    def geometric_process (self, S0, T, num_steps, N):
        """
        N : number of simlations on cols
        num_steps :time on rows
        """
        np.random.seed(123)
        dt = T / num_steps
        S = np.zeros((num_steps + 1, N))
        S [0, :] = S0
        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(int(N / 2))
            Z = np.concatenate((Z, -Z))
            S [t, :] = S [t - 1, :] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + np.sqrt(dt) * self.sigma * Z)
        return S

    # gbm characteristic function
    def ch_function (self, u, T):
        return np.exp(j * u * (self.r - 0.5 * self.sigma ** 2.0) * T - 0.5 * self.sigma ** 2.0 * u ** 2.0 * T)


class Heston_process:
    """
    class for the Heston  process :
    r=risk-free rate cst
    rho=correlation between stock BM and variance BM
    theta=long term mean of the variance process
    kappa=variance of the gamma process
    """

    def __init__ (self, mu, rho, v0, sigma, theta, kappa):
        self.mu = mu
        if sigma < 0 or theta < 0 or kappa < 0 or v0 < 0:
            raise ValueError("sigma , theta , kappa ,v0  must be positive")
        else:
            self.theta = theta
            self.kappa = kappa
            self.sigma = sigma
            self.v0 = v0

        if np.abs(rho) > 1:
            raise ValueError("|rho| must be <1")
        self.rho = rho

    def ch_function (self, u, T):
        i = complex(0.0, 1.0)
        D1 = np.sqrt(
            np.power(self.kappa - self.sigma * self.rho * i * u, 2) + (u * u + i * u) * self.sigma * self.sigma)
        g = (self.kappa - self.sigma * self.rho * i * u - D1) / (self.kappa - self.sigma * self.rho * i * u + D1)
        C = (1.0 - np.exp(-D1 * T)) / (self.sigma * self.sigma * (1.0 - g * D1 * T)) * (
                self.kappa - self.sigma * self.rho * i * u - D1)
        A = self.mu * i * u * T + self.kappa * self.theta * T / self.sigma / self.sigma * (
                self.kappa - self.sigma * self.rho * i * u - D1
        ) - 2 * self.kappa * self.theta / self.sigma / self.sigma * np.log(
            (1 - g * np.exp(-D1 * T)) / (1 - g))
        cf = np.exp(A + C * self.v0)
        return cf
