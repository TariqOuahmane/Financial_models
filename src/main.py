from Process import DiffusionProcess
from Parameters import Option_Parameters
from Black_Sholes_Pricer import BS_pricer

opt_params = Option_Parameters(S0=100, K=100, T=1, exercise="European", payoff="call")
diff_params = DiffusionProcess(r=0.1, sigma=0.2)
BS = BS_pricer(option_info=opt_params, process_info=diff_params)

if __name__ == '__main__':
    print(BS.BS_Price())
    print((BS.MonteCarlo(N=1000000)))
    print(BS.PDE(Nspace=1000,Ntime=1000))
    BS.surface_plot()