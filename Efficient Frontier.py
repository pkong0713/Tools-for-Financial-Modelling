"""
Markowitz's Efficient Frontier - In Modern Portfolio Theory (MPT), EF shows the return vs risk of 
#many randomly generated portfolios, also presenting the mean-variance 'efficient frontier' using Monte-Carlo
#With a risk-free asset, the EF further develops but integrating Capital Allocation Line (Part 4)

#Returns/Rates are [monthly data], as daily data are composed of noises thus giving excess volatility
#Inputs: <timeframe>, <stock_selection>, <rf_df>, <number of generations>"""

#----------------------------------Part 1: Setting up the data--------------------------------------
# Importing the essential libraries
import quandl
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd

# Setting the timeframe
start_date = dt.datetime(2017,1,1)
end_date = dt.datetime(2018,1,1)

# Setting selection of stocks
stock_selection = ["AAPL", "GOOG", "FB", "F", "AMZN", "GOLD"]
stock_price_panel = pd.DataFrame()

# Putting the monthly prices into the panel of stock price
for tickers in stock_selection:
    asset = yf.Ticker(tickers)
    stock_price_panel = stock_price_panel.append(asset.history(period="1y",interval="1mo", start = start_date, end = end_date, auto_adjust = True)['Close'])

# Since it's monthly data, frequency = 12
frequency = 12

stock_price_panel = stock_price_panel.transpose()
stock_price_panel = stock_price_panel.dropna()
stock_price_panel.columns = stock_selection

# Converting them to monthly returns and getting the covariance matrix
stock_return_panel = stock_price_panel.pct_change()
stock_return_panel = stock_return_panel.dropna()
cov_mat = stock_return_panel.cov()
corr_mat = stock_return_panel.corr()

# Converting data from monthly to annual: r_annual = ((1 + r_monthly)^12)-1; cov_annual = cov_monthly * 12 
stock_return_annual = ((1+np.mean(stock_return_panel))**frequency)-1
cov_mat_annual = cov_mat * frequency

#--------------------------------Part 2: Generating the portfolio-----------------------------------
efficient_frontier = pd.DataFrame(columns = ['Return', 'Volatility', 'Sharpe Ratio'])
single_portfolio = pd.DataFrame(columns = ['Return', 'Volatility', 'Sharpe Ratio'])
                             
for tickers in stock_selection:
    single_portfolio[tickers + ' weight'] = []
    efficient_frontier[tickers + ' weight'] = []

# Downlaoding the US-treasury yield (Rf) from Quandl with the timeframe (for Sharpe Ratio)
risk_free_df = quandl.get("USTREASURY/YIELD", authtoken="nr6pF_pcH1fiHoruHFRa", start_date = start_date, end_date = end_date)

# Getting the 3-month rate, and setting the Rf benchmark
rf_df = risk_free_df.iloc[:, 2]
rf = round(rf_df.mean()/100,4)

# Monte-Carlo method of randomly generating portfolios; <#Generations> is parameter
number_of_generations = 5000
portfolio_index = 1

while portfolio_index <= number_of_generations:
    stock_weights = np.random.random(len(stock_selection))
    stock_weights = stock_weights/np.sum(stock_weights)
    stock_weights = stock_weights.reshape(-1,1)
    stock_weights = stock_weights.transpose()
    
    single_portfolio.iloc[:,3:] = stock_weights
    single_portfolio['Return'] = np.dot(stock_weights,stock_return_annual)
    single_portfolio['Volatility'] = np.sqrt(np.dot(stock_weights,np.dot(cov_mat_annual,stock_weights.transpose())))
    single_portfolio['Sharpe Ratio'] = (single_portfolio['Return']-rf)/single_portfolio['Volatility']
    
    efficient_frontier = efficient_frontier.append(single_portfolio)
    
    portfolio_index = portfolio_index + 1

efficient_frontier.reset_index(inplace = True)
efficient_frontier = efficient_frontier.iloc[:,1:]

#------------------------------Part 3: Printing special portfolios----------------------------------
# The two special portfolios are Minimum Vairable Portfolio (MVP), and Maximum Sharpe Ratio (MSR)

# Finding the lowest volatility, then matching it with the row with the lowest vol
min_vol = min(efficient_frontier['Volatility'])
port_mvp = efficient_frontier.loc[efficient_frontier['Volatility'] == min_vol]

# Same method, but for maximum sharpe ratio
max_sharpe = max(efficient_frontier['Sharpe Ratio'])
port_msr = efficient_frontier.loc[efficient_frontier['Sharpe Ratio'] == max_sharpe]

#--------------------------------Part 4: Capital Allocation Line------------------------------------
"""
Investors often will combine a risk-free asset with their risky portfolio in order to reduce risk in their combined portfolio
This gives rise to the 'Captal Allocation Line', where the investor can choose the proportion to invest in their risky portfolio and the risk-free asset
It is given by the equation - CAL: E[rc] = rf + vol_c*(rp-rf)/vol_p, thus having the Sharpe ratio as the slope (increase in excess return per additional unit risk)
Note that E[rc] = y*rp + (1-y)*rf; vol_c = y*vol_p, because vol_rf ~ 0 by definition. Capital Allocation from rf to rc = hedged portfolio, while beyond rc = leveraged """

#Using the max sharpe ratio from the efficient_frontier, using one fund theorem:
rc = max_sharpe* port_msr['Volatility'] + rf
cal = [(0,rf),(port_msr.iloc[0,1],max_sharpe)]
vol_x = [0, port_msr.iloc[0,1], 1.5*port_msr.iloc[0,1]]
return_y = [rf, rc, 1.5*rc]

#------------------------------Part 5: Plotting EF & CAL together-----------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (8,6))
sns.scatterplot(efficient_frontier['Volatility'], efficient_frontier['Return'], s = 10, label = 'Randomly generated portfolios')
plt.scatter(port_mvp['Volatility'], port_mvp['Return'], c = 'Y', marker = '*', s = 100, label = 'Minimum Variance Portfolio')
plt.scatter(port_msr['Volatility'], port_msr['Return'], c = 'R', marker = '*', s = 100, label = 'Maximum Sharpe Ratio')
plt.plot(vol_x,return_y, c = 'R', label = 'Capital Allocation Line')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend(loc=7)
plt.axis(xmin = 0.0)
plt.title("Efficient Frontier with Capital Allocation Line")
plt.show()

#---------------------------------Part 6: Returning the results-------------------------------------
print(f'Max Sharpe Ratio: \n{port_msr.transpose()}'.format())
print('\n')
print(f'Minimum Variance Portfolio: \n{port_mvp.transpose()}'.format())
print('\n')
print(f'With CAL Sharpe Ratio of {round(max_sharpe,2)} and risk-free rate of {rf}.'.format())

