# PLEASE READ BEFORE RUNNING THE CODE!!!
# This code shows you how to construct the binomial tree model and do option pricing
# the code is base on the example in the recitation notes for week 7 (problem 11 & 12)
# one may try to run this code to get exactly the same result as that in the notes.

# You are welcome to modify the code such that it works for American options or for
# binary options (problem 1 in assignment 4), remember to check all the outputs to make
# sure that you are getting the correct output at each step.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 12 * np.log(1.01) # cts-time annual effective interest rate
delta = 0 # cts-time annual effective dividend yield
K = 100 # strike price
T = 2/12 # time to maturity
h = 1/12 # length of each period (in year)
time_limit = int(T / h) # time goes from 0,1,...,time_limit

# Option style
style = "European"

# time_limit has to be an integer
if np.abs(time_limit - T / h) > 1e-4:
    raise Exception('Error! Time limit is not integer!')

# Setting of the binomial tree
S0 = 100 # stock price now
u = 1.04
d = 0.97

# Risk-neutral probability of having a good future state
q = (np.exp((r - delta) * h) - d) / (u - d)
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
#------------------------------------------------------------------------------
# Generate the stock price tree
stock_price = generate_stock_price(S0,u,d,time_limit)
## Print it to test
#print(stock_price)

# Set the payoff for the time at maturity
value = set_payoff(stock_price,K,time_limit)
# Print it to test
##print(value)

# Backward induction to get the price and the replicating portfolio
if style == 'European':
    value, Delta, B = backward_induction_Euro(value,q,h,r,delta,stock_price,u,d,time_limit)
    print('The value of option at each state is:')
    print(value)
    print('For the replicating portfolio:')
    print('The Delta at each state is:')
    print(Delta)
    print('The B at each state is:')
    print(B)

elif style == 'American':
    print('This part is not constructed yet, try to build up the part for the American option on your own!')
