# This code is to solve the problem 4 in assignment 4

import math
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.03 # cts-time annual effective interest rate
delta = 0 # cts-time annual effective dividend yield
# The strike price here is varying
T = 1 # time to maturity
h = 1/15 # length of each period (in year)
time_limit = int(T / h) # time goes from 0,1,...,time_limit

# time_limit has to be an integer
if np.abs(time_limit - T / h) > 1e-4:
    raise Exception('Error! Time limit is not integer!')

# Setting of the binomial tree
S0 = 100 # stock price now
u = 1.01
d = 100/101

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
# Here we assume that the derivative is a put option so the payoff at time_limit
# is max{0,K - S_T}
# Input: binomial tree of stock price and strike price K, time limit
# Output: a dictionary with the value at time_limit already set
def set_payoff(stock_price,K,time_limit):
    value = dict()

    # At the time to maturity
    for num_u in range(time_limit + 1):
        num_d = time_limit - num_u
        ind = 'u' * num_u + 'd' * num_d
        value[ind] = np.max([0,K - stock_price[ind]])
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
            # Remember to also change the payoff function here!!!
            exercise_value = np.max([0,K - stock_price[ind]])

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
# Generate the stock price tree
stock_price = generate_stock_price(S0,u,d,time_limit)
## Print it to test
#print(stock_price)

# For different strike price, do option pricing and plot
# All possible values of K
K_possible_val = np.linspace(85,110,51)

# Record the values of premium and Delta for European and American options
Euro_premium_record = list()
Euro_Delta_record = list()
Amer_premium_record = list()
Amer_Delta_record = list()

# Change strike price for the put option (European)
for K in K_possible_val:
    # Set the payoff for the time at maturity
    value = set_payoff(stock_price,K,time_limit)
    # Print it to test
    ##print(value)

    # Backward induction to get the price and the replicating portfolio
    value, Delta, B = backward_induction_Euro(value,q,h,r,delta,stock_price,u,d,time_limit)

    # Save premium and Delta values
    premium = value['']
    Euro_premium_record.append(premium)
    Delta_init = Delta['']
    Euro_Delta_record.append(Delta_init)

# Change strike price for the put option (American)
for K in K_possible_val:
    # Set the payoff for the time at maturity
    value = set_payoff(stock_price,K,time_limit)
    # Print it to test
    ##print(value)

    # Backward induction to get the price and the replicating portfolio
    value, Delta, B = backward_induction_Amer(value,q,h,r,delta,stock_price,u,d,time_limit)

    # Save premium and Delta values
    premium = value['']
    Amer_premium_record.append(premium)
    Delta_init = Delta['']
    Amer_Delta_record.append(Delta_init)


## Plot respective put premiums
#plt.plot(K_possible_val,Amer_premium_record)
#plt.plot(K_possible_val,Euro_premium_record)
#plt.show()

# Plot the difference of premium
plt.plot(K_possible_val,np.array(Amer_premium_record) - np.array(Euro_premium_record))
## Plot the nontrivial asymptote of the gap curve
#plt.plot(K_possible_val,(1 - np.exp(-r * T)) * K_possible_val - S0 * (np.exp(-delta * T) - 1))
plt.title('Put Premium Gap (American v.s. European) with Different Strike Price')
plt.xlabel('Strike price (K)')
plt.ylabel('Put Premium Gap (P^A - P^E)')
plt.show()
