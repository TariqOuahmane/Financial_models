import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy import sparse


class BS_pricer:
    def __init__ (self, process_info, option_info):
        """
        :param process_info: type of the diffusion model it contain its params
        :param option_info:it contain the option params
        """

        self.r = process_info.r
        self.sigma = process_info.sigma
        self.K = option_info.K
        self.T = option_info.T
        self.S0 = option_info.S0

        self.payoff = option_info.payoff
        self.gbm = process_info.gbm

        self.S_exp = None
        self.price0=None
        self.mesh=None
        self.price = 0
    def payoff_f (self, S):
        if self.payoff == 'call':
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        else:
            raise ValueError("Specify valid payoff")
        return Payoff

    def BS_Price (self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.payoff == "call":
            return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.payoff == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MonteCarlo (self, N: int):
        ST = self.gbm(self.S0, self.T, N)
        payoff = self.payoff_f(ST)
        V = np.exp(-self.r * self.T) * payoff
        return np.mean(V)

    def PDE (self, Nspace, Ntime):
        S_max = 3 * self.K
        S_min = self.K/3

        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretisation
        t ,dt = np.linspace(0, self.T, Ntime, retstep=True)  # Time discretisation

        self.S_exp=np.exp(x)
        payoff = self.payoff_f(np.exp(x))
        # grid discretisation
        V = np.zeros((Nspace, Ntime))
        # boundary terms vector
        B = np.zeros(Nspace - 2)
        if self.payoff=="call" :

            V [-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t[::-1])  # boundary condition
            V [0, :] = 0  # boundary condition
            V [:, -1] = payoff  # terminal condition
        else :
            V[:,-1]=payoff
            V[-1,:]=0
            V[0,:]=self.K*np.exp(-self.r*t[::-1])

        sig2 = self.sigma ** 2
        dxx = dx ** 2
        a = (dt / 2) * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r)
        c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

        for i in range(Ntime - 2, -1, -1):
            B[0] = V[0, i] * a
            B[-1] = V[-1, i] * c
            V[1:-1, i] = spsolve(D, (V[1:-1, i+1] - B))

        # finds the option at S0
        self.price = np.interp(np.log(self.S0), x, V[:, 0])
        self.price0=V[:,0]
        self.mesh=V

        return self.price

    def plot(self):
        plt.plot(self.S_exp,self.payoff_f(self.S_exp),color="red",label="Payoff")
        plt.plot(self.S_exp,self.price0,color="blue",label="Black Sholes Prices")
        plt.xlabel("S")
        plt.ylabel("Price")
        plt.title("Option Black & Sholes price")
        plt.legend()
        plt.show()

    def surface_plot(self):
        fig=plt.figure()
        ax=fig.add_subplot(111,projection="3d")
        X,Y=np.meshgrid(np.linspace(0,self.T,self.mesh.shape[1]),self.S_exp)
        ax.plot_surface(Y,X,self.mesh, cmap=cm.ocean)
        plt.show()
