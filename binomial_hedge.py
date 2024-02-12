from abc import ABC, abstractmethod
from enum import Enum
import math
import random


class PayoffType(str, Enum):
    Call = 'Call'
    Put = 'Put'


class greekType(str, Enum):
    Delta = 'Delta'
    Gamma = 'Gamma'
    Vega = 'Vega'
    Theta = 'Theta'
    Rho = 'Rho'


class EuropeanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return continuation


def crrCalib(r, vol, t):
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1/u, p)

def jrrnCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def jreqCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    return (u, d, 1/2)

def tianCalib(r, vol, t):
    v = math.exp(vol * vol * t)
    u = 0.5 * math.exp(r * t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r * t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def binomialPricer(S, r, vol, trade, n, calib):
    t = trade.expiry / n
    (u, d, p) = calib(r, vol, t)
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff(S * u ** (n - i) * d ** i) for i in range(n + 1)]
    # iterate backward
    for i in range(n - 1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i + 1):
            nodeS = S * u ** (i - j) * d ** j
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j + 1] * (1 - p))
            vs[j] = trade.valueAtNode(t * i, nodeS, continuation)
    return vs[0]


def binomialGreeks(S, r, vol, T, strike, greekType, n, calib):
    trade = EuropeanOption(T, strike, PayoffType.Call)
    if greekType == greekType.Delta:
        dS = S * 0.001
        v_1 = binomialPricer(S + dS, r, vol, trade, n, calib)
        v_2 = binomialPricer(S - dS, r, vol, trade, n, calib)
        return (v_1 - v_2) / (2 * dS)
    elif greekType == greekType.Gamma:
        dS = S * 0.001
        v = binomialPricer(S, r, vol, trade, n, calib)
        v_1 = binomialPricer(S + dS, r, vol, trade, n, calib)
        v_2 = binomialPricer(S - dS, r, vol, trade, n, calib)
        return (v_1 - 2*v + v_2) / (dS ** 2)
    elif greekType == greekType.Vega:
        dSigma = vol * 0.001
        v_1 = binomialPricer(S, r, vol+dSigma, trade, n, calib)
        v_2 = binomialPricer(S, r, vol-dSigma, trade, n, calib)
        return (v_1 - v_2) / (2 * dSigma)
    elif greekType == greekType.Theta:
        dt = 0.004
        trade1 = EuropeanOption(T-dt, strike, PayoffType.Call)
        n1 = int(n*(T-dt)/T)
        v_1 = binomialPricer(S, r, vol, trade1, n1, calib)
        v_2 = binomialPricer(S, r, vol, trade, n, calib) 
        return (v_1 - v_2) / dt
    elif greekType == greekType.Rho:
        dr = 0.0001
        v_1 = binomialPricer(S, r+dr, vol, trade, n, calib)
        v_2 = binomialPricer(S, r-dr, vol, trade, n, calib) 
        return (v_1 - v_2) / (2 * dr) 
    else:
        raise Exception("greekType not supported: ", greekType)
    


def pathSimulation(S, r, vol, current, expiry, stepSize, calib):
    # trade = EuropeanOption(T, strike, PayoffType.Call)
    # t = trade.expiry / n
    (u, d, p) = calib(r, vol, stepSize)
    for i in range():
        rand = random.uniform(0, 1)
        if rand < p:
            S *= u
        else:
            S += d


class HedgingStrategy(ABC):
    def __init__(self):
        super().__init__()


def main():
    S, r, vol, T, TradingDays, strike = 100, 0.03, 0.2, 1, 252, 105
    n = T * TradingDays
    cash = 100
    stock = [0]
    portfolio = [cash]
    stockPrices = [S]

    dt = 1/TradingDays
    (u, d, p) = crrCalib(r, vol, dt)

    for n in range(1, TradingDays*T):
        rand = random.uniform(0, 1)
        if rand < p:
            # print('UP')
            S *= u
        else:
            # print('DOWN')
            S *= d
        # using the wealth equaiton to track the value of portfolio
        stockPrices.append(S)
        delta = binomialGreeks(S, r, vol, n/T, strike, greekType.Delta, n, crrCalib)
        stock.append(stock[-1] + delta)
        x = math.exp(1/252 * r)*portfolio[n-1] + delta * (stockPrices[n] - math.exp(1/252 * r)*stockPrices[n-1])
        portfolio.append(x)
    
    print(S)
    print(f"Payoff of the option: {max(stockPrices[-1] - strike, 0)} ")
    print(f"Delta hedging portfolio: {portfolio[-1]}")
    # print(stockPrices)

main()



