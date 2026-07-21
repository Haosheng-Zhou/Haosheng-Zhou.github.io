# This code is to solve the problem 5 in assignment 4

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
r = 0.05 # cts-time annual effective interest rate
delta = 0 # cts-time annual effective dividend yield
sigma = 0.25 # volatility
K = 100 # The strike price
T = 1 # time to maturity
# length of each period (in year) be specified after fixing the time_limit
# The time_limit here is varying


# Setting of the binomial tree
S0 = 100 # stock price now
# u,d,q are set after knowing h

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Function List
#------------------------------------------------------------------------------
# Finding lower child of a node, return NULL if the node has no child
# Input: the index of the parent, the time limit
# Output: the index of its lower child, all indices are constructed as 'uu..udd..d'
# the number of 'u' means the number of going up and the number of 'd' means the number of going down
def lower_child(p_ind,time_limit):
    if len(p_ind) >= time_limit:
        return NULL

    # Number of u,d in parent node
    p_num_u = sum([1 for i in p_ind if i == 'u'])
    p_num_d = sum([1 for i in p_ind if i == 'd'])
    # Add a 'd'
    c_num_u = p_num_u
    c_num_d = p_num_d + 1

    # Form the index of the parent node
    c_ind = 'u' * c_num_u + 'd' * c_num_d
    return c_ind
#------------------------------------------------------------------------------
# Finding upper child of a node, return NULL if the node has no child
# Input: the index of the parent, the time limit
# Output: the index of its upper child, all indices are constructed as 'uu..udd..d'
# the number of 'u' means the number of going up and the number of 'd' means the number of going down
def upper_child(p_ind,time_limit):
    if len(p_ind) >= time_limit:
        return NULL

    # Number of u,d in parent node
    p_num_u = sum([1 for i in p_ind if i == 'u'])
    p_num_d = sum([1 for i in p_ind if i == 'd'])
    # Add a 'u'
    c_num_u = p_num_u + 1
    c_num_d = p_num_d

    # Form the index of the parent node
    c_ind = 'u' * c_num_u + 'd' * c_num_d
    return c_ind
#------------------------------------------------------------------------------
# Generate a dictionary of stock prices as binary tree
# Input: initial price, u, d, time limit
# Output: a dictionary with the stock price at all times set
def generate_stock_price(S0,u,d,time_limit):
    stock_price = dict()
    stock_price[''] = S0 # Initial price
    for num_u in range(time_limit + 1):
        for num_d in range(time_limit - num_u + 1):
            ind = 'u' * num_u + 'd' * num_d
            stock_price[ind] = S0 * (u ** num_u) * (d ** num_d)
    return stock_price
#------------------------------------------------------------------------------
# The payoff function at time = time_limit, depends on stock price
# Here we assume that the derivative is a call option so the payoff at time_limit
# is max{0,S_T - K}
# Input: binomial tree of stock price and strike price K, time limit
# Output: a dictionary with the value at time_limit already set
def set_payoff(stock_price,K,time_limit):
    value = dict()

    # At the time to maturity
    for num_u in range(time_limit + 1):
        num_d = time_limit - num_u
        ind = 'u' * num_u + 'd' * num_d
        value[ind] = np.max([0,stock_price[ind] - K])
    return value
#------------------------------------------------------------------------------
# Backwardly calculate the value of the option at each time point for European options
# Input: the value dictionary with the value at the maturity date filled in, risk-neutral
# probability q, period length h, interest rate r, dividend yield delta, stock price
# at each state, u, d, time limit
# Output: value dictionary with the value at all time filled in, Delta dictionary
# and B dictionary showing the replicating portfolio at each time point
def backward_induction_Euro(init_value,q,h,r,delta,stock_price,u,d,time_limit):
    # For the value of the option
    value = init_value

    # For the replicating portfolio
    Delta = dict()
    B = dict()

    # Start from 1 period before maturity
    for time in range(time_limit - 1,-1,-1):
        for num_u in range(time + 1):
            num_d = time - num_u
            ind = 'u' * num_u + 'd' * num_d

            # Discounted expected value under risk-neutral probability measure
            low_child_ind = lower_child(ind,time_limit)
            high_child_ind = upper_child(ind,time_limit)
            exp_value = q * value[high_child_ind] + (1 - q) * value[low_child_ind]
            value[ind] = exp_value * np.exp(-r * h)

            # Calculate Delta and B at this time point
            Delta[ind] = np.exp(-delta * h) * (value[high_child_ind] - value[low_child_ind]) / (stock_price[ind] * (u - d))
            # Since Delta * S + B = value
            B[ind] = value[ind] - Delta[ind] * stock_price[ind]
    return value, Delta, B
#------------------------------------------------------------------------------
# Backwardly calculate the value of the option at each time point for American options
# Input: the value dictionary with the value at the maturity date filled in, risk-neutral
# probability q, period length h, interest rate r, dividend yield delta, stock price
# at each state, u, d, time limit
# Output: value dictionary with the value at all time filled in, Delta dictionary
# and B dictionary showing the replicating portfolio at each time point
def backward_induction_Amer(init_value,q,h,r,delta,stock_price,u,d,time_limit):
    # For the value of the option
    value = init_value

    # For the replicating portfolio
    Delta = dict()
    B = dict()

    # Start from 1 period before maturity
    for time in range(time_limit - 1,-1,-1):
        for num_u in range(time + 1):
            num_d = time - num_u
            ind = 'u' * num_u + 'd' * num_d

            # Discounted expected value under risk-neutral probability measure
            # This is the value for holding the option, however, for American option
            # there may exist early exercise
            low_child_ind = lower_child(ind,time_limit)
            high_child_ind = upper_child(ind,time_limit)
            exp_value = q * value[high_child_ind] + (1 - q) * value[low_child_ind]
            holding_value = exp_value * np.exp(-r * h)

            # Compute the payoff exercising call option immediately
            exercise_value = np.max([0,stock_price[ind] - K])

            # Decide whether to exercise early and figure out the value
            value[ind] = np.max([holding_value,exercise_value])
            early_exercise = False
            # When early exercising has its benefits over holding the option
            if np.abs(value[ind] - exercise_value) < 1e-4 and np.abs(value[ind] - holding_value) > 1e-4  :
                early_exercise = True
            #if ind == 'u':
            #    print(holding_value,exercise_value)

            # Calculate Delta and B at this time point only if there's no early exercise
            if early_exercise:
                Delta[ind] = math.nan
                B[ind] = math.nan
                continue

            Delta[ind] = np.exp(-delta * h) * (value[high_child_ind] - value[low_child_ind]) / (stock_price[ind] * (u - d))
            # Since Delta * S + B = value
            B[ind] = value[ind] - Delta[ind] * stock_price[ind]
    return value, Delta, B

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# All possible values of N given in the problem
N_possible_val = [4,8,15,30,60,90,120]
## All possible values of N from 1 to 1000
#N_possible_val = range(4,200)

# Record the values of premium and Delta for European options
premium_record = list()
Delta_record = list()

# Change time limit for the call option (European)
for time_limit in N_possible_val:
    # Length of each period
    h = T / time_limit

    # Binomial tree Parameters
    u = np.exp(sigma * np.sqrt(h))
    d = np.exp(-sigma * np.sqrt(h))

    # Risk-neutral probability of having a good future state
    q = (np.exp((r - delta) * h) - d) / (u - d)

    # Generate the stock price tree
    stock_price = generate_stock_price(S0,u,d,time_limit)
    ## Print it to test
    #print(stock_price)

    # Set the payoff for the time at maturity
    value = set_payoff(stock_price,K,time_limit)
    # Print it to test
    ##print(value)

    # Backward induction to get the price and the replicating portfolio
    value, Delta, B = backward_induction_Euro(value,q,h,r,delta,stock_price,u,d,time_limit)

    # Save premium and Delta values
    premium = value['']
    premium_record.append(premium)
    Delta_init = Delta['']
    Delta_record.append(Delta_init)

# Print the premiums
print('The premiums for those N values:')
print(N_possible_val)
print('are:')
print(premium_record)

## Plot the convergence of premium
#plt.plot(range(4,200),premium_record)
#plt.xlabel('Number of periods (N)')
#plt.ylabel('Call Premium (C)')
#plt.title('Convergence of the Binomial Option Pricing in Continuous Time')
#plt.show()

#------------------------------------------------------------------------------
# Black-Scholes Model option pricing
# Input: init price S0, strike K, interest rate r, ,dividend yield delta,
# time to maturity T, volatility sigma
# Output: the price of the option
def BS_option_pricing(S0,K,r,delta,T,sigma):
    # Compute d_1,d_2
    d1 = (np.log(S0/K) + (r - delta) * T + (sigma ** 2) / (2 * T)) / (sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - delta) * T - (sigma ** 2) / (2 * T)) / (sigma * np.sqrt(T))

    return np.exp(-delta * T) * S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

BS_price = BS_option_pricing(S0,K,r,delta,T,sigma)
print('By B-S formula, the price of this option should be:')
print(BS_price)

#------------------------------------------------------------------------------
# Vary sigma and apply BS formula
sigma_possible_val = [0.2,0.3,0.5]
premium_record = list()

for sigma in sigma_possible_val:
    # Apply BS
    BS_price = BS_option_pricing(S0,K,r,delta,T,sigma)
    premium_record.append(BS_price)

# Print premiums
print(premium_record)
