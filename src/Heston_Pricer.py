from time import time

import scipy.stats as ss
from tqdm import tqdm
import numpy as np
import scipy as scp


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
        # heston parameters

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
        self.payoff = option_info.payoff

    def payoff_f (self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        else:
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def Heston_paths (self, N, paths):
        """
        N:number of steps
        paths: number of simulated paths
        return : value of S an V at maturity T
        """
        v_T = np.zeros(paths)  # value of v at T
        S_T = np.zeros(paths)  # value of S at T
        v = np.zeros(N)
        S = np.zeros(N)
        dt = self.T / (N - 1)
        dt_sq = np.sqrt(dt)
        for path in tqdm(range(paths)):
            # generate random BMs
            W_s_arr = np.random.normal(loc=0, scale=1, size=N - 1)
            W_v_arr = self.rho * W_s_arr + np.sqrt(1 - self.rho ** 2) * np.random.normal(loc=0, scale=1, size=N - 1)
            S [0] = self.S0  # stock at 0
            v [0] = self.v0  # variance at 0

            for t in tqdm(range(0, N - 1)):
                v [t + 1] = max(
                    v [t] + self.kappa * (self.theta - v [t]) * dt + self.sigma * np.sqrt(v [t]) * W_v_arr [
                        t] * dt_sq,0.0)
                S [t + 1] = S [t] * np.exp((self.r - v [t] * 0.5) + np.sqrt(v [t]) * dt_sq * W_s_arr [t])

            S_T [path] = S [N - 1]
            v_T [path] = v [N - 1]

        return S_T, v_T

    def Heston_log_paths (self, N, paths):
        X0 = np.log(self.S0)
        Y0 = np.log(self.v0)
        T_vec, dt = np.linspace(0, self.T, N, retstep=True)
        dt_sq = np.sqrt(dt)
        # feller condition
        assert 2 * self.kappa * self.theta > self.sigma ** 2
        # genrate random BM
        Mu = np.array([0.0, 0.0])
        cov = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs(mean=Mu, cov=cov, size=(N - 1, paths))
        W_S = W [:, :, 0]
        W_v = W [:, :, 1]
        # vectors
        Y = np.zeros((N, paths))
        Y [0, :] = Y0
        X = np.zeros((N, paths))
        X [0, :] = X0
        v = np.zeros(N)
        # generate paths
        for t in tqdm(range(0, N - 1)):
            v = np.exp(Y [:, t])
            v_sq = np.sqrt(v)
            Y [t + 1, :] = (
                    Y [t, :] + (1 / v) * (self.kappa * (self.theta - v) - 0.5 * self.sigma ** 2) * dt + self.sigma * (
                    1 / v_sq) * dt_sq * W_v [t, :]
            )
            X [t + 1, :] = X [t, :] + (self.r - 0.5 * v) * dt + v_sq * dt_sq * W_S [t, :]
        return np.exp(X), np.exp(Y)

    def Fourier_Inverse (self):
        """
        price obtained by inversion of the ch function
        """
        pass

    def IV_Lewis (self):
        """
        implied volatility from the lewis formula
        """
        pass

    def Monte_Carlo (self, N, paths, Err=False, Time=False):
        """
        Heston MC
        N=time steps
        paths=number of simulated paths
        Err=standard error if True
        time = execution time if True
        """
        t_init = time()
        S_T, v_T = self.Heston_paths(N=N, paths=paths)
        # S_T = S_T.reshape((paths, 1))
        DiscountedPayoff = np.exp(-self.r * self.T) * self.payoff_f(S_T)
        V = np.mean(DiscountedPayoff)
        std_error = scp.stats.sem(DiscountedPayoff)
        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, std_error, elapsed
            else:
                return V, std_error
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V
