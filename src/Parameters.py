

class Option_param :
    """
    class to define the params of the option S0,K,r,T, and v0 optional (spot variance)
    """
    def __init__(self,S0,K,T,v0,payoff="call",exercise="European"):
        self.S0=S0
        self.K=K
        self.T=T
        self.v0=v0
        if exercise=="European" or exercise=="American" :
            self.exercise=exercise
        else :
            raise ValueError("invalide exercise type")
        if payoff=="call" or payoff=="put" :
            self.payoff=payoff
        else :
            raise ValueError("invalid payoff type .")