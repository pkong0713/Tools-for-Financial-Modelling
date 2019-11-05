"""
CAPM Modelling - In CAPM, required return of an asset = Rf + Beta * MRP,
Where Beta = Cov(Ra, Rm)/Var(Rm), and MRP = Expected return on market - risk-free rate

Returns are [monthly data], as daily data are composed of noises thus giving excess volatility
Mean of [3-month] [US-Treasury] over the period is chosen as the risk-free rate
Inputs: <timeframe>, <rf_panel>, <market>, <asset>"""

#--------------------------------Part 1: Setting up the model----------------------------------------
# Importing the necessary libraries
import quandl
import datetime as dt
import yfinance as yf
import numpy as np

# Setting the timeframe
start_date = dt.datetime(2015,1,1)
end_date = dt.datetime(2018,1,1)

# Downlaoding the US-treasury yield (Rf) from Quandl with the timeframe
risk_free_rates = quandl.get("USTREASURY/YIELD", authtoken="nr6pF_pcH1fiHoruHFRa", start_date = start_date, end_date = end_date)

# Getting the 3-month rate, and setting the Rf benchmark(Chosen using [:,2])
rf_df = risk_free_rates.iloc[:, 2]
rf = round(rf_df.mean()/100,4)

# Downloading S&P 500 data (Benchmark) from yfinance
market = "^GSPC"
benchmark_market = yf.Ticker(market)

# Getting expected return on market for Benchmark (SPY)
benchmark_price = benchmark_market.history(interval="1mo", start = start_date, end = end_date, auto_adjust=True)['Close']
benchmark_price_pct = benchmark_price.pct_change()
benchmark_price_pct = benchmark_price_pct.dropna()
rm = benchmark_price[-1]/benchmark_price[0]-1

# Getting risk for Benchmark (SPY)
std_market = np.std(benchmark_price)

# Getting return for Apple stock, "AAPL"
ticker = "AAPL"
asset = yf.Ticker(ticker)
return_on_asset = asset.history(interval="1mo", start = start_date, end = end_date, auto_adjust = True)['Close']
return_on_asset = return_on_asset.dropna()
return_on_asset_pct = return_on_asset.pct_change()
return_on_asset_pct = return_on_asset_pct.dropna()
ra = return_on_asset[-1]/return_on_asset[0]-1

# Getting risk for Asset
std_asset = np.std(return_on_asset)

# Getting the Market Risk Premium & Variance of Benchmark
maket_risk_premium = rm - rf
var_market = np.var(benchmark_price)

# Getting the covariance and computing Beta_A
cov_ab = np.cov(return_on_asset_pct,benchmark_price_pct)
beta_a = round(cov_ab[0][1]/cov_ab[1][1],4)

# Alpha based on the mean values
alpha = round(np.mean(return_on_asset_pct)-beta_a*np.mean(benchmark_price_pct),4)

# Finally the CAPM
expected_return_on_asset = rf + beta_a * (maket_risk_premium)

print(f'For the ticker {ticker} vs benchmark {market},'.format())
print(f'Beta of {ticker} is {beta_a} from cov/var, and'.format())
print(f"CAPM Expected return = {expected_return_on_asset.round(4)} vs Actual return = {ra.round(4)}.".format())

#---------------------------------Part 2: Graphing the model-----------------------------------------
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

X = benchmark_price.pct_change().values[1:]
X = X.reshape(-1,1)

y = return_on_asset.pct_change().values[1:]

regressor.fit(X,y)

style.use('ggplot')
plt.title('Finding Beta from graphs')
plt.xlabel('Market return $R_m$')
plt.ylabel('Stock return $R_a$')
plt.scatter(benchmark_price.pct_change(),return_on_asset.pct_change(), c='blue', marker = 'x')
plt.plot(X ,regressor.predict(X), c = 'red')
plt.text(0.015, 0.0195, r'$R_a = \beta * R_m + \alpha$', fontsize=12)
plt.grid(True)

print(f'Model coefficient (Beta) from plot is {regressor.coef_.round(4)}, and the intercept (Alpha) is {regressor.intercept_.round(4)}.'.format())

#----------------------------------Part 3: Statistical Parts-----------------------------------------
cor_ab = round(cov_ab[0][1]/((cov_ab[0][0]**(1/2))*(cov_ab[1][1]**(1/2))),4)
r2_ab = round(cor_ab**2,4)

print(f'Correlation between {ticker} and {market} is {cor_ab}, and R^2 is {r2_ab}.'.format())
