import unittest
from src.Option_Info import Option_Info
from src.Diffusion_Processes import GeometricBM
from src.BS_pricer import BS_pricer
import numpy as np

class Test_option_info(unittest.TestCase) :
    def test_payoff(self):
        option1=Option_Info(S0=0,K=0,T=1,v0=0,payoff="call",exercise="European")
        self.assertEqual(option1.payoff_value(S=np.array([0])),np.array([0]))
        option2=Option_Info(S0=10,K=10,T=1,v0=0,payoff="call",exercise="European")
        self.assertEqual(option2.payoff_value(S=np.array([10])), np.array([0]))

    def test_BS_price(self):
        option1=Option_Info(S0=0,K=10,T=1,v0=0,payoff="call",exercise="European")
        process1=GeometricBM(r=0,sigma=0.1,mu=0)
        BS=BS_pricer(option_info=option1,process_info=process1)
        self.assertEqual(BS.BlackSholesPrice(),0)