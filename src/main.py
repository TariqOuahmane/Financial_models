import numpy as np

from BS_pricer import BS_pricer
from Parameters import Option_param
from Processes import Diffusion_Process , Heston_process
from Heston_Pricer import Heston_pricer
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rho = -0.8  # correlation coefficient
    kappa = 3  # mean reversion coefficient
    theta = 0.1  # long-term mean of the variance
    sigma = 0.25  # (Vol of Vol) - Volatility of instantaneous variance
    T = 0.1  # Terminal time
    v0 = 0.08  # spot variance
    S0 = 100  # spot stock price
    r = 0.1
    K = 101
    # Black-Scholes model testing
    option_info=Option_param(S0=S0,K=K,T=T,v0=0.04,exercise="European",payoff="call")
    diffusion_params=Diffusion_Process(r=r,sigma=sigma,mu=r)
    heston_params=Heston_process(sigma=sigma,theta=1,mu=r,rho=rho,kappa=kappa)
    BSM = BS_pricer(option_info,diffusion_params)
    HP=Heston_pricer(option_info=option_info,process_info=heston_params)

    print("by the black and sholes formula : ",BSM.BlackSholesPrice())
    print("Monte carlo from the BS pricer : ",BSM.MonteCarlo(N=1000000,steps=100))
    print("COS methdod ", BSM.COS_method(N=100000,L=1))
    
