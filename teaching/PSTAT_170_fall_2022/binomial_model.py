# This code is to estimate u and d in the binomial tree stock price model
# To run the code for AMC, change the name inside the pd.read_csv() function
# AND change the which_delta to AMC_delta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

# Load data and parameters (only use the data in 2020)
price = pd.read_csv('GS.csv')
price = price.loc[price['Date'] > '2020-01-01']

# Annual Dividend yield
GS_delta = 0.0276
AMC_delta = 0
which_delta = GS_delta #<-- which company we are estimating
# Turn into continuous-time annual effective dividend yield
delta = np.log(1 + which_delta)

# Risk-free interest rate (taken as 2022/11/01), 3 month term annual effective
interest_rate = 0.0423
# Turn into continuous-time annual effective interest rate
r = np.log(1 + interest_rate)

# Monthly time step
h = 1/12

#------------------------------------------------------------------------------
# Estimate \sigma, the volatility
# Calculate daily log-returns log(S_{t+1}/S_t)
price['shifted_Close'] = np.roll(price['Close'],1)
price['quotient'] = price['Close'] / price['shifted_Close']
price['log_return'] = np.log(price['quotient'])

# Compute standard deviation of log-return
# Eliminate the first entry
log_ret = price['log_return'][2:]
log_ret_std = stat.stdev(log_ret)
# Scale to work as annualized volatility
volatility = log_ret_std * np.sqrt(price.shape[0] - 1)

# Compute upward and downward probability using (10.9) in textbook
u = np.exp((r - delta) * h + volatility * np.sqrt(h))
d = np.exp((r - delta) * h - volatility * np.sqrt(h))

# Print result
print('Estimated u is ' + str(u) + ', estimated d is ' + str(d))
