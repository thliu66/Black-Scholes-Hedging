from abc import ABC, abstractmethod
from enum import Enum
import math
import numpy as np
import pandas as pd
import scipy.stats as ss


class PayoffType(str, Enum):
    Call = 'Call'
    Put = 'Put'


class Direction(str, Enum):
    Long = 'Long'
    Short = 'Short'


class Portfolio:
    pass

    
    
class PathsGeneration(ABC):
    def __init__(self, currentVal, currentTime, steps, expiry, interestRate):
        self.currentVal = currentVal
        self.currentTime = currentTime
        self.steps = steps
        self.expiry = expiry
        self.interestRate = interestRate
        self.stepSize = expiry/steps
        
    @abstractmethod
    def progress_to_t(self):
        pass

    @abstractmethod
    def simulate_path(self):
        pass


class HedgingStrategy(ABC):
    def __init__(self, direction, payoffType, spot, strike, time_to_expiry):
        self.direction = direction
        self.payoffType = payoffType
        self.spot = spot
        self.strike = strike
        self.time_to_expiry = time_to_expiry

    @abstractmethod
    def rebalance(self):
        pass
    

class TradingSimulator:
    def __init__(self, cash, PathsGeneration, HedgingStrategy):
        self.PnL = cash
        self.PathsGeneration = PathsGeneration
        self.HedgingStrategy = HedgingStrategy
        self.book = pd.DataFrame(columns=['Time', 'Price', 'Delta', 'Total Delta Position',
                                          'Adjustment Cashflow', 'Interest on Adjustments'])
        

    def trading(self):
        row = [0, self.PathsGeneration.currentVal, self.HedgingStrategy.rebalance(), None, None, None]
        self.book.loc[len(self.book)] = row
        while self.PathsGeneration.currentTime < self.PathsGeneration.expiry:
            row = [None for _ in range(6)]
            # update the classes with newest information
            self.PathsGeneration.progress_to_t()
            spot_price = self.PathsGeneration.currentVal
            self.HedgingStrategy.spot = spot_price
            self.HedgingStrategy.time_to_expiry = round(self.PathsGeneration.expiry - self.PathsGeneration.currentTime, 4)

            # calculate the portfolio adjustment
            stock_rebalance = self.HedgingStrategy.rebalance()

            row[0] = self.PathsGeneration.currentTime # Time
            row[1] = spot_price #Price
            row[2] = stock_rebalance # Delta
            # row[3] = the change of delta
            row[4] = -stock_rebalance * spot_price # Adjustment Cashflow
            # row[5] = interest generated by 'Adjustment Cashflow'
            self.book.loc[len(self.book)] = row

        self.book['Total Delta Position'] = self.book['Delta'] - self.book['Delta'].shift(1)
        self.book['Interest on Adjustments'] = self.book.apply(lambda x: x['Adjustment Cashflow']*self.PathsGeneration.interestRate*x['Time'], 
                                                               axis=1)

    def ProfitAndLost(self):
        return self.PnL
            

class DeltaHedging(HedgingStrategy):
    def __init__(self, direction, payoffType, spot, strike, time_to_expiry, interest_rate, volatility):
        super().__init__(direction, payoffType, spot, strike, time_to_expiry)
        self.interest_rate = interest_rate
        self.volatility = volatility
    
    # can use the greeks engine written
    def rebalance(self):
        if self.time_to_expiry < 1e-6:
            d_1 = float('inf')
        else:
            d_1 = ((math.log(self.spot / self.strike) + (self.interest_rate + self.volatility ** 2 / 2) * self.time_to_expiry) 
                / (self.volatility * math.sqrt (self.time_to_expiry)))
        if self.payoffType == PayoffType.Call:
            if self.direction == Direction.Long:
                return ss.norm.cdf(d_1)
            elif self.direction == Direction.short:
                return -ss.norm.cdf(d_1)
            else:
                raise Exception(f"Invalid direction: {self.direction}")
        
        if self.payoffType == PayoffType.Put:
                if self.direction == Direction.Long:
                    return ss.norm.cdf(d_1) - 1
                elif self.direction == Direction.short:
                    return - (ss.norm.cdf(d_1) - 1)
                else:
                    raise Exception(f"Invalid direction: {self.direction}")
        else:
            raise Exception(f"The payoff type {self.payoffType} is not supported")

    
class GeometricBrownianMotion(PathsGeneration):
    def __init__(self, currentVal, currentTime, steps, expiry, interestRate, volatility):
        super().__init__(currentVal, currentTime, steps, expiry, interestRate)
        self.drift = interestRate
        self.volatility = volatility
        self.times = []
        self.prices = []
    
    def progress_to_t(self):
        self.times.append(self.currentTime)
        self.prices.append(self.currentVal)
        change = ((self.drift - 0.5 * self.volatility ** 2) * self.stepSize 
                + self.volatility * np.random.normal(0, math.sqrt(self.stepSize)))
        self.currentVal = math.exp(math.log(self.currentVal) + change)
        self.currentTime += self.stepSize

        # print(f"Time is {self.currentTime}, value is {self.currentVal}") 
    def simulate_path(self):
        while self.currentTime < self.expiry:
            self.progress_to_t()


if __name__ == "__main__":
    portfolio = {'cash': 1, 'call': 0, 'stock': 0}
    initial_wealth = 100 
    stock_spot = 100
    strike = 110
    interest_rate = 0.01
    stock_volatility = 0.27
    time_to_expiry = 1
    N = 50
    currentTime = 0

    strategy = DeltaHedging(Direction.Long, PayoffType.Call, 
                            stock_spot, strike, time_to_expiry, interest_rate, stock_volatility)
    process = GeometricBrownianMotion(stock_spot, currentTime, N, time_to_expiry, interest_rate, stock_volatility)
    test = TradingSimulator(initial_wealth, process, strategy)
    test.trading()
    print(test.book)

    # print(f"The PnL for delta hedging strategy is {test.PnL}")
    # process = GeometricBrownianMotion(stock_spot, currentTime, N, expiry, drift, stock_volatility, interest_rate)
    # payoff = max(test.PathsGeneration.currentVal - strike, 0)
    # print(f"The payoff of the call option is {payoff}")

    # plot the paths of simulated stock prices 
    # import matplotlib.pyplot as plt
    # time_prices = pd.DataFrame(columns=['Time', 'Price'])
    # for _ in range(50):
    #     process = GeometricBrownianMotion(stock_spot, currentTime, N, expiry, interest_rate, stock_volatility)
    #     process.simulate_path()
    #     x = process.times
    #     y = process.prices
    #     plt.plot(x, y)
    # plt.xlabel("Time")
    # plt.ylabel("Log Simulated Price")
    # plt.legend()
    # plt.title('Geometric Brownian Motion')
    # plt.show()
