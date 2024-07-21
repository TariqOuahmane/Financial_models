

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

class Option_Parameters:

    def __init__(self,S0,K,T,payoff,exercise):
        self.S0=S0
        self.K=K
        self.T=T

        if exercise=="European" or exercise=="American" :
            self.exercise = exercise
        else :
            raise ValueError("specify a valid exercise type")

        if payoff=="call" or payoff=="put":
            self.payoff = payoff
        else :
            raise ValueError("Specify a valid payoff type ")


