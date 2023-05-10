import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setting up the normal distributino for the price change
mu, sigma = 0, 1
number_days = 1000
seed = np.random.seed(0)
# Use seed 0 for downward trend, 1 for upward trend
daily_p_change = np.random.normal(mu,sigma, number_days)

#creating a historic price path
starting_price = 100.00
historic_price = [starting_price]
for i in range(len(daily_p_change)):
    historic_price.append(round(historic_price[i] + (historic_price[i]*daily_p_change[i]/100),2))

#plt.plot(historic_price)
#number_bars = int(number_days/50)
#plt.hist(daily_p_change,bins=number_bars)

borders = []
number_states = 2

percentiles = []
for i in range(number_states+1):
    percentiles.append(int(100*i/number_states))

slices = []
for i in range(len(percentiles)):
    slices.append(np.percentile(daily_p_change,percentiles[i]))
    


