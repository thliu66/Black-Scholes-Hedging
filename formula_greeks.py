import numpy as np
import math
import scipy.stats as ss

def main():
    pass

class BlackScholesVanillaGreeksEngine:
    def __init__(self, S, K, T, sigma, r):
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r
        self.d_1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt (T))
        self.d_2 = self.d_1 - self.sigma * math.sqrt(self.T)

    def call_delta(self):
        # spot derivative
        return ss.norm.cdf(self.d_1)
    
    def call_gamma(self):
        # 2nd spot derivative
        return norm_pdf(self.S) / (self.S * self.sigma * math.sqrt(self.T))
    
    def call_vega(self):
        # volatility derivative
        return self.S * math.sqrt(self.T) * norm_pdf(self.d_1)

    def call_rho(self):
        # r derivative
        return self.r * self.K * math.exp(-self.r * self.T) * ss.norm.cdf(self.d_2)

    def call_theta(self):
        # negative of T derivative
        a = -self.S * norm_pdf(self.d_1) * self.sigma / (2 * math.sqrt(self.T))
        b = self.r * self.K * math.exp(-self.r * self.T) * ss.norm.cdf(self.d_2)


def norm_pdf(x):
    return math.exp(-x**2 /2) / (math.sqrt(2 * math.pi))

