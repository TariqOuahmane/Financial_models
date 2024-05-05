


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


