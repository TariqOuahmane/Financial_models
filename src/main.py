import matplotlib.pyplot as plt
import numpy as np

from BS_pricer import BS_pricer
from Option_Info import Option_Info
from Diffusion_Processes import GeometricBM , Heston_process
from Heston_Pricer import Heston_pricer
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    r = 0.06  # drift
    rho = -0.8  # correlation coefficient
    kappa = 3  # mean reversion coefficient
    vbar = 0.1  # long-term mean of the variance
    gamma = 0.2 # (Vol of Vol) - Volatility of instantaneous variance
    T= 3 # Terminal time
    K = 1.10  # Stike
    v0 = 0.06 # spot variance
    S0 = 1.0 # spot stock price
    div=0.06


    # Black-Scholes model testing
    option_info=Option_Info(S0=S0,K=K,T=T,v0=v0,exercise="American",payoff="put")
    diffusion_params=GeometricBM(r=r, sigma=gamma, mu=r)
    heston_params=Heston_process(sigma=gamma,theta=vbar,mu=r,rho=rho,kappa=kappa,v0=v0)
    BSM = BS_pricer(option_info,diffusion_params)
    HP=Heston_pricer(option_info=option_info,process_info=heston_params)
    strikes=np.arange(50,151,5)

    print("american MC 1 ",BSM.AMC(Nsteps=100,Nsim=100000,degree=5))
    print("by the black and sholes formula : ",BSM.BlackSholesPrice())
    print("Monte carlo from the BS pricer : ",BSM.MonteCarlo(N=100000,steps=100))
    print("Price using B&S PDE ",BSM.PDE(Nspace=100000,Ntime=100))
    print("COS methdod B&S model", BSM.COS_method(N=10000,L=3))
    print("COS method heston model",HP.Option_price_COS_method(N=10000,L=3))

