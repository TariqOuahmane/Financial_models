import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve, splu
from scipy.stats import norm
from scipy import sparse
import scipy.stats as ss

from src.COS_method import COSMethod


class BS_pricer:

    def __init__ (self, option_info, process_info):
        """
        :param option_info: it contains (S,K,T)
        :param process_info: type of the diffusion process it contain (r,mu,sigma)
        """
        self.S_exp = None
        self.r = process_info.mu
        self.sigma = process_info.sigma
        self.S0 = option_info.S0
        self.K = option_info.K
        self.T = option_info.T
        self.geometric_process = process_info.geometric_process
        self.payoff_value = option_info.payoff_value  # the option payoff type
        self.payoff_type = option_info.payoff_type
        self.exercise = option_info.exercise
        self.ch_function = process_info.ch_function

        self.cos_obj = COSMethod(option_info=option_info, process_info=process_info)

    def BlackSholesPrice (self):
        """
        Calculates price for call option according to the formula.
        Formula: S*N(d1) - PresentValue(K)*N(d2)
        """
        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be
        # exercised)

        assert self.T>0 and self.K>0

        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of
        # the option)
        d2 = (np.log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.payoff_type == 'call':
            return self.S0 * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)
        elif self.payoff_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MonteCarlo (self, N: int, steps: int):
        S = self.geometric_process(S0=self.S0, N=N, T=self.T, num_steps=steps)
        payoff = self.payoff_value(S [-1])
        V = np.sum(payoff) / N
        return np.exp(-self.r * self.T) * V

    # cos method
    def COS_method (self, N: int, L: int):
        return self.cos_obj.COS_Price(N=N, L=L, ch_function=self.ch_function)

    def PDE (self, Nspace: int, Ntime: int):
        S_max = 3 * self.K
        S_min = self.K / 3

        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretisation
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # Time discretisation

        self.S_exp = np.exp(x)
        payoff = self.payoff_value(self.S_exp)
        # grid discretisation
        V = np.zeros((Nspace, Ntime))
        # boundary terms vector
        B = np.zeros(Nspace - 2)
        if self.payoff_type == "call":

            V [-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t [::-1])  # boundary condition
            V [0, :] = 0  # boundary condition
            V [:, -1] = payoff  # terminal condition
        else:
            V [:, -1] = payoff
            V [-1, :] = 0
            V [0, :] = payoff [0] * np.exp(-self.r * t [::-1])

        sig2 = self.sigma ** 2
        dxx = dx ** 2
        a = (dt / 2) * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r)
        c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)
        for i in range(Ntime - 2, -1, -1):
            B [0] = V [0, i] * a
            B [-1] = V [-1, i] * c
            V [1:-1, i] = DD.solve(V [1:-1, i + 1] - B)

        # finds the option at S0
        self.price = np.interp(np.log(self.S0), x, V [:, 0])
        self.price0 = V [:, 0]
        self.mesh = V

        return self.price

    def AMC (self, Nsteps: int, Nsim: int, degree: int):
        if self.exercise != "American":
            raise ValueError("This method is reserved for american type exercises")
        gbm = self.geometric_process(S0=self.S0, T=self.T, num_steps=Nsteps, N=Nsim)
        option_payoff = self.payoff_value(S=gbm)
        matrix_values = np.zeros_like(option_payoff)
        matrix_values [-1, :] = option_payoff [-1, :]
        time_unit = self.T / Nsteps
        df = np.exp(-self.r * time_unit)
        for t in range(Nsteps - 1, 0, -1):
            good_paths = option_payoff [t, :] > 0
            regression = np.polyfit(gbm [t, good_paths], matrix_values [t + 1, good_paths] * df, deg=degree)
            continuation_value = np.polyval(regression, gbm [t, good_paths])
            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise [good_paths] = option_payoff [t, good_paths] > continuation_value
            matrix_values [t, exercise] = option_payoff [t, exercise]
            matrix_values [t + 1:, exercise] = 0
            discount_paths = matrix_values [t, :] == 0
            matrix_values [t, discount_paths] = matrix_values [t + 1, discount_paths] * df
        return np.mean(matrix_values [1, :]) * df

    def plot (self):
        plt.plot(self.S_exp, self.payoff_value(self.S_exp), color="red", label="Payoff")
        plt.plot(self.S_exp, self.price0, color="blue", label="B&S price by PDE ")
        plt.xlabel("S")
        plt.ylabel("Price")
        plt.title("Option Black & Sholes price")
        plt.legend()
        plt.show()

    def surface_plot (self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape [1]), self.S_exp)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        plt.show()
