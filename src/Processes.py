import numpy as np
from scipy.stats import norm

j=complex(0.0,1.0)
class Diffusion_Process:
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

    def geometric_process(self,S0,T,num_steps,N):
        dt=T/num_steps
        S=np.zeros((num_steps,N))
        S[0]=S0
        for t in range(1,num_steps) :
            Z=np.random.normal(0,1,N)
            S[t]=S[t-1]*np.exp((self.r - 0.5 * self.sigma ** 2) * dt+ np.sqrt(dt) * self.sigma*Z)
        return S

    def ch_f (self, u,T):
        return np.exp(j * u * (self.r - 0.5 * self.sigma ** 2.0) * T - 0.5 * self.sigma ** 2.0 * u ** 2.0 * T)


class Heston_process:
    """
    class for the Heston  process :
    r=risk free rate cst
    rho=correlation beteween stock BM and variance BM
    theta=long term mean of the variance process
    kappa=variance of the gamma process
    """
    def __init__(self,mu,rho,sigma,theta,kappa) :
        self.mu=mu
        if sigma < 0 or theta<0 or kappa<0:
            raise ValueError("sigma , theta , kappa  must be positive")
        else :
            self.theta = theta
            self.kappa = kappa
            self.sigma=sigma

        if np.abs(rho)> 1 :
            raise ValueError("|rho| must be <1")
        self.rho=rho

        #feller condition
