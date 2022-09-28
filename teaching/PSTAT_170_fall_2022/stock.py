# This code is to compute mean return and volatility of the stock price

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

# Load data
price = pd.read_csv('GS.csv')

# Calculate daily return
price['return'] = (price['Close'] - price['Open']) / price['Open']
daily_mean_ret_arith = np.mean(price['return'])
daily_mean_ret_geom = (np.prod(price['return'] + 1)) ** (1/price.shape[0]) - 1
daily_volatility = stat.stdev(price['return'])

# Plot stock price
plt.plot(price['Date'],price['Open'])
plt.xlabel('Trading day')
plt.ylabel('Stock price')
plt.show()

# Plot (After-hours trading)
plt.plot(price['Date'],price['return'])
plt.xlabel('Trading day')
plt.ylabel('Daily return')
plt.show()

# Two versions of mean return, arithmetic v.s. geometric
print('The daily mean return is: ' + str(daily_mean_ret_arith))
print('The daily mean return (geometric) is: ' + str(daily_mean_ret_geom))
print('The daily volatility (std-dev) is: ' + str(daily_volatility))

#-------------------------------------------------------------------------------
# Term transformed into year
yearly_return_l = list()
for year in range(2014,2021):
    part_price = price.loc[price['Date'] >= str(year) + '-01-01']
    part_price = part_price[part_price['Date'] <= str(year) + '-12-31']
    yearly_open = part_price['Open'].iloc[0]
    yearly_close = part_price['Close'].iloc[-1]
    # Compute yearly return
    yearly_return = (yearly_close - yearly_open) / yearly_open
    yearly_return_l.append(yearly_return)

yearly_mean_ret_arith = np.mean(yearly_return_l)
yearly_mean_ret_geom = (np.prod(np.array(yearly_return_l) + 1)) ** (1/len(yearly_return_l)) - 1
yearly_volatility = stat.stdev(yearly_return_l)

# Plot
plt.plot(range(2014,2021),yearly_return_l)
plt.xlabel('Year')
plt.ylabel('Yearly return')
plt.show()

# Two versions of mean return, arithmetic v.s. geometric
print('The yearly mean return is: ' + str(yearly_mean_ret_arith))
print('The yearly mean return (geometric) is: ' + str(yearly_mean_ret_geom))
print('The yearly volatility (std-dev) is: ' + str(yearly_volatility))
