from abc import ABC, abstractmethod
from enum import Enum
import math
import numpy as np
import scipy.stats as ss

from formula_greeks import BlackScholesVanillaGreeksEngine


class PayoffType(str, Enum):
    Call = 'Call'
    Put = 'Put'


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
        if continuation == None:
            return self.payoff(S)
        else:
            return continuation
        

# class Portfolio:
#     def __init__(self):
#         self.asset = []
#         self.account = {}

#     def addAsset(self, asset):
#         if asset not in self.account.values():
#             self.asset.append(asset)
#         else:
#             raise Exception(f"The asset {asset} is already in your account")
    
    


class PathsGeneration(ABC):
    def __init__(self, currentVal, currentTime, steps, expiry):
        self.currentVal = currentVal
        self.currentTime = currentTime
        self.steps = steps
        self.expiry = expiry
        self.stepSize = expiry/steps

    @abstractmethod
    def path_at_t(self):
        pass


class HedgingStrategy(ABC):
    @abstractmethod
    def rebalance(self):
        pass

    @abstractmethod
    def updateSpotPrice(self, price):
        pass
    

class TradingSimulator:
    def __init__(self, cash):
        self.PnL = cash
        self.book = {}
        

    def trading(self, PathsGeneration, HedgingStrategy):
        while PathsGeneration.currentTime < PathsGeneration.expiry:
            spot_price = PathsGeneration.path_at_t()
            stock_position = HedgingStrategy.rebalance
            self.PnL -= spot_price * stock_position
            self.book[stock_position] = self.book[stock_position].get(stock_position, 0) + spot_price
            DeltaHedging.updateSpotPrice(spot_price)
        
        for position, price in self.book.item():
            self.PnL += position * price 
        
        return self.PnL
            

class DeltaHedging(HedgingStrategy):
    def __init__(self, spot, strike, expiry, interest_rate, volatility):
        self.spot = spot
        self.strike = strike
        self.expiry = expiry
        self.interest_rate = interest_rate
        self.volatility = volatility
        self.d_1 = ((math.log(self.spot / self.strike) + (self.interest_rate + self.volatility ** 2 / 2) * expiry) 
                    / (volatility * math.sqrt (expiry)))

    def delta(self):
         # spot derivative
        return ss.norm.cdf(self.d_1)
    
    def rebalance(self):
        return ss.norm.cdf(self.d_1)
    
    def updateSpotPrice(self, price):
        self.spot = price


class GeometricBrownianMotion(PathsGeneration):
    def __init__(self, currentVal, currentTime, steps, expiry, drift, volatility, interest_rate):
        super().__init__(currentVal, currentTime, steps, expiry)
        self.drift = drift
        self.volatility = volatility
        self.interest_rate = interest_rate
    
    def path_at_t(self):
        change = ((self.interest_rate - 0.5 * self.volatility ** 2) * self.stepSize 
                + self.volatility * math.sqrt(self.stepSize) * np.random.normal(0, 1, 1))
        self.currentVal = math.exp(math.log(self.currentVal) + change)
        self.currentTime += self.stepSize

        return self.currentVal, self.currentTime



def main():
    portfolio = {'cash': 1, 'call': 0, 'stock': 0}
    initial_wealth = 100 
    stock_spot = 100
    strike = 110
    interest_rate = 0.05
    stock_volatility = 0.27
    expiry = 0.25
    drift = interest_rate
    N = 50
    currentTime = 0

    strategy = DeltaHedging(stock_spot, strike, expiry, interest_rate, stock_volatility)
    process = GeometricBrownianMotion(stock_spot, currentTime, N, expiry, drift, stock_volatility, interest_rate)
    test = TradingSimulator(100)
    PnL = test.trading(PathsGeneration=process,
                       HedgingStrategy=strategy)
    print(f"The PnL for delta hedging strategy is {PnL}")

main()



