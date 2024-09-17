import numpy as np


class COSMethod:
    def __init__ (self, process_info, option_info):
        self.r = process_info.mu
        self.sigma = process_info.sigma
        self.S0 = option_info.S0
        self.K = option_info.K
        self.T = option_info.T

        self.payoff_type = option_info.payoff_type

    def __call__ (self, N, L, ch_function):
        return self.COS_Price(N, L, ch_function)

    def COS_Price (self, N: int, L: int, ch_function):
        j = complex(0.0, 1.0)
        a = -L * np.sqrt(self.T)
        b = L * np.sqrt(self.T)
        k = np.linspace(0, N - 1, N)
        u = k * np.pi / (b - a)
        x0 = np.log(self.S0 / self.K)
        H_k = self.CallPutCoefficients(a, b, k)  # the coeff for the option price
        temp = ch_function(u, self.T) * H_k
        temp [0] = temp [0] * 0.5
        mat = np.exp(j * np.dot((x0 - a), u))
        OptionVal = np.exp(-self.r * self.T) * np.real(temp.dot(mat)) * self.K
        return OptionVal

    def CallPutCoefficients (self, a, b, k):
        H_k = 0.0
        if self.payoff_type == "call":
            c = 0.0
            d = b
            if a < b < 0.0:
                H_k = np.zeros(len(k))
            else:
                H_k = 2.0 / (b - a) * (self.chi(a, b, c, d, k) - self.psi(a, b, c, d, k))

        if self.payoff_type == "put":
            c = a
            d = 0.0
            H_k = 2.0 / (b - a) * (-self.chi(a, b, c, d, k) + self.psi(a, b, c, d, k))

        return H_k

    # the characteristic function og GBM

    @staticmethod
    def chi (a, b, c, d, k):
        u = k * np.pi / (b - a)
        chi = 1.0 / (1 + u ** 2)
        expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                           * (c - a) / (b - a)) * np.exp(c)
        expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                             (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                               * np.pi * (c - a) / (
                                                                                                       b - a)) * np.exp(
            c)
        chi = chi * (expr1 + expr2)
        return chi

    @staticmethod
    def psi (a, b, c, d, k):
        psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
        psi [1:] = psi [1:] * (b - a) / (np.pi * k [1:])
        psi [0] = d - c
        return psi
