import pandas as pd
import numpy as np
import scipy.stats as stats
import math
from numpy.linalg import inv

import matplotlib.pyplot as plt

from scipy.stats import norm

from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compound(r):
    return (r+1).cumprod()-1

def semi_deviation(returns: pd.Series):
    negative_returns = returns[returns < 0]
    semi_dev = negative_returns.std(ddof=0)
    return semi_dev

def drawdown(returns: pd.Series):
    
    wealth = 1000*(1+returns).cumprod()
    peaks = wealth.cummax()

    dataframe = pd.DataFrame({
        'Returns':returns,
        'Wealth_index':wealth,
        'Previous_peaks':peaks
    })

    return dataframe

def drawdown_comp(returns: pd.Series, plot= False):
    
    wealth = 1000*(1+returns).cumprod()
    peaks = wealth.cummax()

    drawdown = (wealth - peaks)/peaks

    if plot:
        drawdown.plot()
        x_pos = drawdown.index[int(len(drawdown) * 0.65)]
        drawdown_text = f'Drawdown: {drawdown.min().round(4)}\nDate: {drawdown.idxmin()}'
        plt.text(y=drawdown.min(), x=x_pos, s=drawdown_text)

        return plt.show()

    else:
        return drawdown

def max_drawdown(returns: pd.Series):
    wealth = 1000*(1+returns).cumprod()
    peaks = wealth.cummax()

    drawdown = (wealth - peaks)/peaks

    return drawdown.min()

def skewness(values: pd.Series):
    demeaned_val = values - values.mean()
    dvcube = demeaned_val**3
    expected_dvcube = dvcube.mean()

    std = values.std(ddof=0)
    stdcube = std**3

    skewness = expected_dvcube/stdcube

    return skewness

def kurtosis(values: pd.Series):
    demeaned = values - values.mean()
    demeanedquad = demeaned**4
    demeanedquad_mean = demeanedquad.mean()

    std = values.std(ddof=0)
    std_quad = std**4

    kurtosis = demeanedquad_mean/std_quad

    return kurtosis

def jarque_bera(values, level: float = 0.05):
    
    if isinstance(values,pd.DataFrame):
        values.aggregate(jarque_bera)
    else:
        stat, pvalue = stats.jarque_bera(values)
        name = values.name

        if pvalue < level:
            print(f'{name}\nJB_stat:{stat:.4f}\np_value:{pvalue:.4f}\nNull Hypothesis is rejected, Not normal\n')
        else:
            print(f'{name}\nJB_stat:{stat:.4f}\np_value:{pvalue:.4f}\nNull Hypothesis is NOT rejectes, Normal\n')

def historic_VaR(returns, level):

    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(historic_VaR, level = level)
    
    elif isinstance(returns, pd.Series):
        return np.percentile(returns, level*100)

    else:
        raise TypeError("Introduce Dataframe or Series data")

def Gauss_VaR(returns, level, modified = False):
    
    if modified:
        if isinstance(returns, pd.DataFrame):
            return returns.aggregate(Gauss_VaR,level = level, modified = modified)
        
        elif isinstance(returns, pd.Series):
            z = norm.ppf(level)
            k = kurtosis(returns)
            s = skewness(returns)
            cf_z = (z + 
                    1/6*(z**2-1)*s +
                    1/24*(z**3-3*z)*(k-3) -
                    1/36*(2*z**3-5*z)*s**2
                    )
            
            VaR = returns.mean()+cf_z*returns.std(ddof=0)

            return VaR
        
        else:
            raise TypeError("Insert a Dataframe or Series type")

    else:
        if isinstance(returns, pd.DataFrame):
            return returns.aggregate(Gauss_VaR,level = level)
        
        elif isinstance(returns, pd.Series):
            z = norm.ppf(level)
            VaR = returns.mean() + z*returns.std(ddof=0)
            return VaR
        
        else:
            raise TypeError("Insert a Dataframe or Series type")
    
def CVaR(returns, level, method = historic_VaR):

    methods = {
        'historic': historic_VaR,
        'gauss': Gauss_VaR,
        'modified': lambda r, l: Gauss_VaR(r, l, modified=True)
    }

    if isinstance(method, str):
        method = methods[method]

    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(CVaR, level=level, method=method)
    
    elif isinstance(returns, pd.Series):
        below_var = returns <= method(returns, level) # Method is replaced by the function assigned to the string in methods dict
        return returns[below_var].mean() # Here u use arithmetic mean, because you're trying to get the expected return across mutiple "independent" returns
    
    else:
        raise TypeError("Introduce DataFrame or Series data") 
    
def annual_vol(r, periods_per_year):
    annual_v = r.std()*(periods_per_year**0.5)

    return annual_v

def annual_return(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def sharpe_ratio(r, risk_free, periods_per_year): # The riskfree is yearly

    rf = (risk_free+1)**(1/periods_per_year)-1
    
    difference = r - rf

    annual_dif = annual_return(difference, periods_per_year) #This sharpe ratio is annualized
    annual_v = annual_vol(r, periods_per_year)

    sharpe_rat = annual_dif/annual_v

    return sharpe_rat

def portfolio_return(weights, returns):
    '''
    Annualized portfolio's return
    '''

    return weights.T @ returns #Returns must be the annualized form

def portfolio_vol(weights, covariance_matrix):
    '''
    Annualized portfolio's volatility
    '''

    return (weights.T @ covariance_matrix @ weights) ** 0.5

def minimize_vol(target_return, returns, covar):
    '''
    Calculation of Minimum volatility for a Target_return 
    '''

    n = returns.shape[0] # Number of assets
    init_guess = np.repeat(1/n,n) # Starting point for optimizer

    # Constraints
    bounds = ((0,1),)*n # Bounds for each asset

    return_is_target = {
        'type': 'eq',
        'args': (returns,),
        'fun':  lambda weights, returns: target_return - portfolio_return(weights, returns)
    } # Constraints is met if the portfolio return is the same target return specified

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    } # Constraint is met if the portfolio's weight are the same

    results = minimize(portfolio_vol, init_guess, args= (covar,), method= 'SLSQP',
                       options={'disp': False},
                       constraints= (return_is_target, weights_sum_to_1),
                       bounds= bounds)
    
    return results.x

def optimal_weights(n_points, annual_ret, covar):
    '''
    Get the optimal weights for each target return
    '''

    target_returns = np.linspace(annual_ret.min(),annual_ret.max(),n_points)
    weights = [minimize_vol(tr, annual_ret, covar) for tr in target_returns]

    return weights

def maximize_sharpe(riskfree_rate, returns, covar):
    '''
    Return the weights combination that results in the Maximum Sharpe for a Target_return 
    '''

    n = returns.shape[0] # Number of assets
    init_guess = np.repeat(1/n,n) # Starting point for optimizer

    # Constraints
    bounds = ((0.0,1.0),)*n # Bounds for each asset

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    } # Constraint is met if the portfolio's weight sum 1

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        
        return -(r - riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess, # We look out for minimum negative sharpe ratio, because this built-in function only let you minimize
                       args= (riskfree_rate, returns, covar), 
                       method= 'SLSQP',
                       options={'disp': False},
                       constraints= (weights_sum_to_1,),
                       bounds= bounds)
    
    return pd.Series(results.x, index= covar.index)

def lowest_volatility(covar):
    '''
    Return the weights of the portfolio with the lowest volatility
    '''
    n_assets = covar.shape[0] 

    return maximize_sharpe(0,np.repeat(1,n_assets),covar) # Trick: you turn Sharpe ratio into "1-0/vol". The unique variable is volatility. So the minimizer algorithm can only reduce volatility

def efficient_front_plot(n_points, annual_ret, covar, cml = False, riskfree_asset= 0, show_ewp= False, show_gmv= False):
    '''
    Plot the efficient frontier
    '''
    weights = optimal_weights(n_points, annual_ret, covar) # This trace the Efficient frontier

    target_returns = np.linspace(annual_ret.min(),annual_ret.max(),n_points) # Weight optimizer will use target return to catch the best combination for each of them

    # Calculation of portfolio returns and volatility for each combination of weights
    weighted_returns = [portfolio_return(w,annual_ret) for w in weights]
    weighted_volts = [portfolio_vol(w, covar) for w in weights]

    # Plotting
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x = weighted_volts, y = weighted_returns, linewidths= 0.5)
    ax.plot(weighted_volts, weighted_returns, color= 'Red')

    # Show GMV portfolio as a dot
    if show_gmv:
        gmv = lowest_volatility(covar)

        gmv_x = portfolio_vol(gmv, covar)
        gmv_y = portfolio_return(gmv, annual_ret)

        ax.plot(gmv_x, gmv_y, color= 'Green', marker= 'o')

    # Show Equal Weight Portfolio as a dot
    if show_ewp:
        ewp = np.repeat(1/annual_ret.shape[0],annual_ret.shape[0])

        ewp_x = portfolio_vol(ewp, covar)
        ewp_y = portfolio_return(ewp, annual_ret)
        
        ax.plot(ewp_x, ewp_y, color= 'Red', marker= "o")

    # Add CML
    if cml:
        msr = maximize_sharpe(riskfree_asset, annual_ret, covar)

        msr_y= portfolio_return(msr,annual_ret)
        msr_x= portfolio_vol(msr,covar)

        ax.set_xlim(left=0)

        cml_x = [0, msr_x]
        cml_y = [riskfree_asset, msr_y]

        ax.plot(cml_x, cml_y, color ='Blue',marker="o")
    
    return ax

def run_cppi(risky_r, safe_r= None, m= 3, start= 1000, floor= 0.8, riskfree_rate= 0.03, drawdown= None): # Zero coupon represent liabilities
    dates = risky_r.index #Risky_r should be entered with a date index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak= account_value

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r) #This is a way to get risk free returns with the same shape of risky assets
        safe_r[:] = riskfree_rate/12

    # This is to get an historical spread of your amount in your portfolio
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    safe_w_history = pd.DataFrame().reindex_like(risky_r)
    floor_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps): #All values get updated here
        # Updating the floor
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w,1)
        risky_w = np.maximum(risky_w,0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # Updating the account value
        account_value = (risky_alloc*(1+risky_r.iloc[step])) + (safe_alloc*(1+safe_r.iloc[step]))

        # Save the values, so I can look at the history
        cushion_history.iloc[step]= cushion
        risky_w_history.iloc[step]= risky_w
        account_history.iloc[step]= account_value
        safe_w_history.iloc[step]= safe_w
        floor_history.iloc[step]= floor_value
    
    risky_wealth = (1+risky_r).cumprod()*start

    backtest_result= {
        'Wealth': account_history,
        'Risky wealth': risky_wealth,
        'Risky budget': cushion_history,
        'Risky allocation': risky_w_history,
        'Floor history': floor_history,
        'Safe allocation': safe_w_history,
    
        'M': m,
        'Start': start,
        'Floor': floor,
        'Risky_r': risky_r,
        'Safe_r': safe_r
    }

    return backtest_result

def summary_stats(r, freq, risk_free_rate= 0.03): # Only use the return change (pct_change)
    ann_r = annual_return(r,freq)
    ann_vol = annual_vol(r,freq)
    skwns = skewness(r)
    krts = kurtosis(r)
    Var_cornish= Gauss_VaR(r,0.05, True)
    historic_CVaR = CVaR(r,0.05, method= 'historic')
    sharpe_rat = sharpe_ratio(r, risk_free_rate, freq)
    drawdowns = max_drawdown(r)

    summary= pd.DataFrame({
        'Annual return': ann_r,
        'Annual volatility': ann_vol,
        'Skewness': skwns,
        'Kurtosis': krts,
        'Cornish-Fisher VaR (5%)': Var_cornish,
        'Historical CVaR': historic_CVaR,
        'Sharpe Ratio': sharpe_rat,
        'Max Drawdown': drawdowns
    })
    return summary

def gbm(n_years=10, n_scenarios= 1000, mu= 0.07, sigma= 0.15, steps_per_year= 12, s_0= 100.0, prices= True):
    '''
    Evolution of Stock Price by using GBM
    '''
    dt = 1/steps_per_year # Periods 
    n_steps = n_years * steps_per_year +1
    rets_plus_1 = np.random.normal(loc= (1+mu*dt), scale= (sigma*np.sqrt(dt)),size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    rets_plus_1 = pd.DataFrame(rets_plus_1)

    # Return prices
    outcome = s_0*rets_plus_1.cumprod() if prices else rets_plus_1-1

    return outcome

def gbm_plot(n_years=10, n_scenarios= 1000, mu= 0.07, sigma= 0.15, steps_per_year= 12, s_0= 100.0, prices= True):
    '''
    Evolution of Stock Price plot by using GBM 
    '''
    dt = 1/steps_per_year # Periods 
    n_steps = n_years * steps_per_year
    rets_plus_1 = np.random.normal(loc= (1+mu*dt), scale= (sigma*np.sqrt(dt)),size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    rets_plus_1 = pd.DataFrame(rets_plus_1)

    # Return prices
    outcome = s_0*rets_plus_1.cumprod() if prices else rets_plus_1-1

    ax = outcome.plot(legend= False, color= 'Blue', alpha= 0.2, figsize= (12,5))
    ax.axhline(s_0, ls=':', color= 'Black')
    return 

def gbm_cppi(n_scenarios= 10, mu= 0.07, sigma= 0.15, m= 3, start= 100, floor= 0.8, riskfree_rate= 0.03, y_max= 100):
    ret = gbm(n_scenarios= n_scenarios, mu=mu, sigma= sigma, steps_per_year= 12, prices= False)
    ret = pd.DataFrame(ret)
    
    cppi = run_cppi(ret, riskfree_rate= riskfree_rate, floor= floor, start= start, m= m)
    wealth = cppi['Wealth']

    y_max= wealth.values.max()*y_max/100
    terminal_wealth= wealth.iloc[-1]

    # Violations
    violations_mask= terminal_wealth<start*floor
    n_violations= violations_mask.sum()
    prop_violations= n_violations/n_scenarios*100
    e_shortfall= np.dot(terminal_wealth-start*floor, violations_mask)/n_violations if n_violations > 0 else 0.0

    # Metrics of Final return
    wealth_mean= terminal_wealth.mean()
    wealth_median= terminal_wealth.median()

    # Graph
    fig, ax= plt.subplots(1,2, sharey= True, gridspec_kw={'width_ratios':[3,2]},figsize= (20,5))
    plt.subplots_adjust(wspace=0)

    # Figure 1: Line plot of return simulations
    wealth.plot(ax= ax[0], legend= False, alpha= 0.2, color= 'Green')
    ax[0].set_ylim(top= y_max)
    ax[0].axhline(y= start, ls= ':', color= 'Black')
    ax[0].axhline(y= start*floor, ls= '--', color= 'Red')

    # Figure 2: Histogram of end values 
    terminal_wealth.plot.hist(ax=ax[1], ec= 'w', bins= 70, orientation= 'horizontal', color= 'Green')
    ax[1].axhline(y= start, ls= ':', color= 'Black')
    ax[1].text(transform= ax[1].transAxes, x= 0.80, y= 0.80, 
               s= 'Metrics of Final wealth', 
               ha= 'left', va= 'center')
    ax[1].text(transform= ax[1].transAxes, x= 0.80, y= 0.70, 
               s= f'Mean: {wealth_mean:.2f}\nMedian: {wealth_median:.2f}', 
               ha= 'left', va= 'center')
    if n_violations > 0:
        ax[1].text(transform= ax[1].transAxes, x= 0.80, y= 0.60, 
                s= f'Vilations: {n_violations} ({prop_violations:.2f}%)\nShortfall: {e_shortfall:.2f}$', 
                ha= 'left', va= 'center')

# Discount
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

# Present Value
def present_val(values, r):
    '''
    Function to compute the ammount Present Value of a list
    '''
    dates= values.index
    disc = discount(dates, r)

    return disc.multiply(values,axis=0).sum()

# Funding Ratio
def funding_ratio(assets, liabilities, r):
    '''
    Calculates funding ratio based on present value of Assets and liabilities
    '''

    return present_val(assets, r)/present_val(liabilities, r)

# Annualized to short term rate
def ann_to_inst(r):
    '''
    Converts annualiazed rates to a short term rate
    '''

    return np.log1p(r)

# Short term rate to annualized
def inst_to_ann(r):
    '''
    Converts short term rates to an annualized rate
    '''

    return np.expm1(r)

# CIR
def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

# Bond Cash flows
def bond_cash_flows(maturity, principal= 100, coupon_rate= 0.03, coupons_per_year= 12):
    n_coupons= round(maturity*coupons_per_year)
    coupon_amnt= principal*coupon_rate/coupons_per_year
    coupon_times= np.arange(1,n_coupons+1)

    # Here you create the cash flows
    cash_flows = pd.Series(data=coupon_amnt, index= coupon_times)

    # Here you get the last cash flow plus the principal
    cash_flows.iloc[-1] += principal
    
    return cash_flows

# Bond price
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t]) # The maturity decreases t/coupons per year each iteration and Disc_rate move forward as "t" increases
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return present_val(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(cash_flow, discount_rate):
    disc= discount(cash_flow.index, discount_rate)*pd.DataFrame(cash_flow) # Here you get cash flow discounted
    w_flows= disc/disc.sum()

    return np.average(cash_flow.index, weights= w_flows.iloc[:,0])

def matching_duration(cf_t,cf_s,cf_l, discount_rate):
    '''
    Calculate the weight of Short term bond by Macaulay duration metric
    '''
    md_l= macaulay_duration(cf_l, discount_rate)
    md_s= macaulay_duration(cf_s, discount_rate)
    md_t= macaulay_duration(cf_t, discount_rate)

    return (md_l - md_t)/(md_l - md_s)

# Bond's total returns
def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and fixed coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons= pd.DataFrame().reindex_like(monthly_prices)
    t_max= monthly_prices.index.max()

    coupon_date= np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype= int)

    coupons.iloc[coupon_date]= principal*coupon_rate/coupons_per_year
    total_returns= (monthly_prices+coupons)/monthly_prices.shift()-1

    return total_returns.dropna()

# Mix allocator
def fixedmix_allocator(r1, r2, w1, **kwargs):
    '''
    Produces a time series over T steps of allocaations between PSP and GHP across N scenarios
    PHP and GHP are T x N Dataframe, in which T is time stamp and N is the scnearios
    Returns an T x N Dataframe of PSP weights
    '''
    return pd.DataFrame(data=w1, index= r1.index, columns= r1.columns)

# Glide allocator
def glidepath_allocator(r1, r2, start_glide= 1, end_glide= 0):
    '''
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    '''
    # Target-Date-Fund = Investment plan that changes as your goals approach
    n_points= r1.shape[0]
    n_col= r1.shape[1]
    path= pd.Series(data= np.linspace(start_glide, end_glide, n_points)) # Here you create a series that reduced itself by linspaces function
    paths= pd.concat([path]*n_col, axis=1)
    paths.index= r1.index
    paths.columns= r1.columns

    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    '''
    Allocate between psp and ghp to lift thi up in order to
    stay away from the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns the PSP weight with the same shape of psp/ghp
    '''
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP_r and ZC_prices must have the same shape')
    
    n_steps, n_scenarios = psp_r.shape
    account_value= np.repeat(1,n_scenarios)
    floor_value= np.repeat(1, n_scenarios)
    w_history= pd.DataFrame(index= psp_r.index, columns= psp_r.columns)

    for step in range(n_steps):
        floor_value= floor*zc_prices.iloc[step] # PV of Floor assuming today's rates and flat YC. Here you can get a floor value higher than account value
        cushion= (account_value-floor_value)/account_value # Proportion. If floor value got a value higher than account value, cushion becomes negative
        psp_w= (m*cushion).clip(0,1) # If cushion got negative, psp_w would be zero by clip trick jiji
        ghp_w= 1-psp_w
        psp_alloc= account_value*psp_w
        ghp_alloc= account_value*ghp_w

        # recompute the new account value at the end of this step
        account_value= psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step]= psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, ddmax, m=3):
    '''
    Allocate between psp and ghp to lift thi up in order to
    stay away from the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns the PSP weight with the same shape of psp/ghp
    '''
    if ghp_r.shape != psp_r.shape:
        raise ValueError('PSP_r and GHP_r must have the same shape')
    
    n_steps, n_scenarios = psp_r.shape
    account_value= np.repeat(1,n_scenarios)
    floor_value= np.repeat(1, n_scenarios)
    peak_value= np.repeat(1, n_scenarios)
    w_history= pd.DataFrame(index= psp_r.index, columns= psp_r.columns)

    for step in range(n_steps):
        floor_value= (1-ddmax)*peak_value # PV of Floor assuming today's rates and flat YC. Here you can get a floor value higher than account value
        cushion= (account_value-floor_value)/account_value # Proportion. If floor value got a value higher than account value, cushion becomes negative
        psp_w= (m*cushion).clip(0,1) # If cushion got negative, psp_w would be zero by clip trick jiji
        ghp_w= 1-psp_w
        psp_alloc= account_value*psp_w
        ghp_alloc= account_value*ghp_w

        # recompute the new account value and max value at the end of this step
        account_value= psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        peak_value= np.maximum(peak_value, account_value)
        w_history.iloc[step]= psp_w
    return w_history

# Mixer of two return DataFrames
def bt_mix(r1, r2, allocator, **kwargs):
    '''
    Runs a back test that allocate tow sets of returns
    The returns ar T x N, in which T is time stamp and N is the scnearios
    Allocator is a function that takes the two sets of returns and brings back weights for each of them by each period
    It returns a DataFrame of returns in N scenarios
    '''

    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 must have the same shape')
    
    weights = allocator(r1, r2, **kwargs)

    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    
    r_mix= weights*r1+(1-weights)*r2
    return r_mix

# Terminal values
def terminal_values(rets):
    return (rets+1).prod()

# Summary terminal stats
def terminal_stats(rets, floor= 0.8, cap= np.inf, name='Stats'):
    '''
    Summarize the stats of the final flow according to a T x N returns Dataframe
    It calculate the metrics for each N scenario and then average to get one value
    It returns a DataFrame
    '''
    # Floor represents the goal (GHP) or simply the minimum value u don't want returns fall down abroad
    # Cap is the limit of return risings
    terminal_wealth= (rets+1).prod()
    breach= terminal_wealth < floor
    reach= terminal_wealth >= cap
    p_breach= breach.mean() if breach.sum() > 0 else np.nan
    p_reach= reach.mean() if reach.sum() > 0 else np.nan
    e_short= (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus= (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats= pd.DataFrame.from_dict({
        'Mean': terminal_wealth.mean(),
        'Std': terminal_wealth.std(),
        'Breach_probability': p_breach,
        'Reach_probability': p_reach,
        'Avg_shortfall': e_short,
        'Avg_surplus': e_surplus
    }, orient= 'index', columns=[name])

    return sum_stats

# Style analysis

def tracking_error(r_a, r_b):
    '''
    Returns the Tracking Error between the two return series
    r_a: Reference, r_b: Benchmark
    '''

    return np.sqrt(((r_a-r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):
    '''
    Return the tracking error between the reference returns and
    a portfolio of building block returnsheld with given weights
    '''
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dependent_variable, explanatory_variables):
    '''
    Return optimal weights that minimizes the tracking error between
    a portfolio of the explantory variables and the dependent variable
    dependent_var: Portfolio whose style you want to discover
    explanatory_var: Return of benchmark assets classes (Large cap, small cap, index, more)
    '''
    n= explanatory_variables.shape[1]
    init_guess= np.repeat(1/n,n) # Starting point (equal weighted)
    bounds= ((0.0, 1.0),)*n # an N-tuples of 2 tuples

    # Constraints
    weights_sum_to_1= {'type':'eq',
                       'fun': lambda weights: np.sum(weights) - 1}

    # The goal is to find the portfolio weights that best replicate the returns of a target portfolio.
    solution= minimize(portfolio_tracking_error, init_guess,
                       args= (dependent_variable, explanatory_variables,), method= 'SLSQP',
                       constraints= (weights_sum_to_1,),
                       bounds= bounds)
    
    weights = pd.Series(solution.x, index= explanatory_variables.columns)
    return weights

# Covariances

def sample_cov(r, **kwargs):
    '''
    Returns the sample covariance of the supplied returns
    '''
    return r.cov()

def cc_cov(r, **kwargs):
    '''
    Estimate covariance matrix by using Elton/Gruber constant correlation model
    '''
    rhos= r.corr()
    n= rhos.shape[0]

    # Calculate the mean
    rhos_mean= (rhos.values.sum()-n)/(n*(n-1)) # Exclude the diagonals
    ccor= np.full_like(rhos, rhos_mean)
    np.fill_diagonal(ccor,1.)
    sd= r.std()
    ccov= ccor * np.outer(sd,sd) # Outer: [[sd1 * sd1, sd1 * sd2, ...],[sd2 * sd1, sd2 * sd2, ...], ...]
    return pd.DataFrame(ccov, index= r.columns, columns= r.columns)

def pca_covariance(returns, n_components=3):
    """
    Estimate a denoised covariance matrix using a PCA-based statistical
    factor model. Instead of the raw sample covariance (which is noisy
    with short estimation windows), this decomposes returns into a few
    systematic factors (top principal components) + idiosyncratic noise,
    and reconstructs a cleaner covariance matrix from that.

    returns      : DataFrame of asset returns (rows = time, cols = assets)
    n_components : number of principal components to treat as "systematic"
                   risk factors. Rule of thumb: check pca.explained_variance_ratio_
                   and pick enough components to explain ~70-90% of variance.

    Returns
    -------
    cov_matrix : DataFrame (assets x assets) — denoised covariance estimate,
                 usable anywhere you'd plug in a sample covariance matrix
                 (e.g. as Sigma in your Black-Litterman model).
    """
    X = returns.values

    # Standardize each asset's returns to mean 0, std 1.
    # This puts PCA on a correlation basis rather than covariance basis,
    # so no single high-volatility asset dominates the component structure.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA on the standardized returns, keep only the top n_components
    # (these represent the dominant systematic risk factors).
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Reconstruct the "systematic" correlation matrix from the top components.
    # pca.components_ has shape (n_components, n_assets); pca.explained_variance_
    # gives the variance captured by each component.
    factor_corr = pca.components_.T @ np.diag(pca.explained_variance_) @ pca.components_

    # Idiosyncratic (residual) variance = what's left after removing the
    # systematic components, per asset — this is the "noise" PCA didn't explain.
    reconstructed = pca.inverse_transform(pca.transform(X_scaled))
    residual_var_scaled = np.var(X_scaled - reconstructed, axis=0)

    # Denoised correlation matrix = systematic part + idiosyncratic part on the diagonal
    corr_matrix = factor_corr + np.diag(residual_var_scaled)

    # Convert back from standardized (correlation) scale to covariance scale
    # by rescaling with each asset's original standard deviation.
    std = scaler.scale_  # per-asset std devs computed by StandardScaler
    cov_matrix = corr_matrix * np.outer(std, std)

    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample

# Weighting methods for backtesting

def weight_ew(r, cap_w=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_w is not None:
        cw = cap_w.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_w, series= True,**kwargs):
    '''
    Returns the weights of the CW portfolio based on the time series of capweights
    '''
    if series:
        w= cap_w.loc[r.index[1]]

        return w/w.sum()

    else:
        return cap_w

def weight_gmv(r, cov_estimator= sample_cov, **kwargs):
    '''
    Produces the weights of the GMV portfolio given a covariance matrix of returns
    '''
    est_cov = cov_estimator(r, **kwargs)
    
    return lowest_volatility(est_cov)

# Backtesting

def backtest_ws(r, estimation_window=60, weighting=weight_ew, rebalance_freq=1, **kwargs):
    """
    Like backtest_ws, but only recomputes weights every `rebalance_freq` periods
    instead of every single period. Between rebalances, weights are held constant.

    r               : asset returns
    estimation_window : lookback window used to estimate params at each rebalance
    weighting       : function(r_window, **kwargs) -> weights vector
    rebalance_freq  : how many periods between rebalances (e.g. ~21 = monthly,
                       ~42 = every 2 months, if r is daily/business-day data)
    """
    n_periods = r.shape[0]

    # Only step forward by rebalance_freq instead of by 1
    windows = [(start, start + estimation_window)
               for start in range(0, n_periods - estimation_window, rebalance_freq)]

    rebal_dates = [r.index[win[1]] for win in windows]
    weights_at_rebal = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    weights_at_rebal = pd.DataFrame(weights_at_rebal, index=rebal_dates, columns=r.columns)

    # Hold weights constant between rebalance dates
    full_index = r.iloc[estimation_window:].index
    weights = weights_at_rebal.reindex(full_index).ffill()

    returns = (weights * r).sum(axis="columns", min_count=1)
    return returns

def backtest_ws_passive(r, estimation_window=60, weighting= weight_ew, **kwargs):
    """
    True buy-and-hold / passive backtest: weights are set ONCE using the first
    `estimation_window` rows, and never rebalanced again. Weights drift on
    their own as assets compound at different rates — no trading afterward.

    r                 : asset returns
    estimation_window : lookback window used ONLY at the very start
    weighting         : function(r_window, **kwargs) -> initial weights vector
    """
    w0 = weighting(r.iloc[:estimation_window], **kwargs)   # one weight vector, computed once

    r_oos = r.iloc[estimation_window:]
    growth = (1 + r_oos).cumprod()                          # cumulative growth per asset
    asset_value = growth * w0                                # value of each sleeve, starting at w0_i
    portfolio_value = asset_value.sum(axis="columns")         # total value, starts at 1

    portfolio_value_prev = portfolio_value.shift(1).fillna(1.0)
    returns = portfolio_value / portfolio_value_prev - 1

    drifted_weights = asset_value.div(portfolio_value, axis=0)  # for inspection only

    return returns, drifted_weights

# Black litterman model

def as_colvec(x):
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)

def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)

def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
    Computes the posterior expected returns based on 
    the original black litterman reference model

    W.prior must be an N x 1 vector of weights, a Series
    Sigma.prior is an N x N covariance matrix, a DataFrame
    P must be a K x N matrix linking Q and the Assets, a DataFrame
    Q must be an K x 1 vector of views, a Series
    Omega must be a K x K matrix a DataFrame, or None
    if Omega is None, we assume it is proportional to variance of the prior
    delta and tau are scalars
    """

    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)

def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

# Another options for weighting a portfolio using previous parameters calculated

def weight_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w

def weight_star(delta, sigma, mu):
    '''
    Minimize the difference between your views and the market equilibrium
    '''

    return (inverse(sigma).dot(mu))/delta

# Risk contribution

def risk_cont(weights, cov):

    if isinstance(weights, pd.DataFrame):
        rc= weights.agg(risk_cont, cov= cov)
        return rc
    
    else:
        sigma_w= weights @ cov
        std_p= portfolio_vol(weights, cov)**2

        rc= weights * sigma_w / std_p

        return rc

# Risk parity portfolio

def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_cont(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

# Weighting method using previous parameters calculated

def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    weights= target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)

    return pd.Series(weights, index= cov.columns)

# Weighting method for backtesting

def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)

def weight_bl(r, p, q, cap_w, cov_estimator=sample_cov,
                  confidence_lvl=0.5, tau=0.025, rf=0.05, portfolio='msr', **kwargs):
    """
    One BL step, callable per rolling window by backtest_ws.
    r             : window of returns (estimation_window rows x N assets)
    p             : K x N view matrix (fixed across windows)
    q             : K x 1 view vector, same periodicity as r (fixed)
    cap_w         : N x 1 market-cap weights
    cov_estimator : function(r, **kwargs) -> covariance DataFrame
                    e.g. m1.sample_cov, m1.cc_cov, m1.shrinkage_cov
    portfolio     : 'msr', 'gmv', or 'erc'
    """
    sigma_prior = cov_estimator(r, **kwargs)

    omega = proportional_prior(sigma_prior, tau=tau, p=p)
    multiplier = (1 - confidence_lvl) / confidence_lvl
    omega = omega * multiplier

    delta = calculate_delta(r, cap_w, rf)

    P = p.reset_index(drop=True)
    Q = q.reset_index(drop=True)
    Omega = omega.to_numpy()

    bl_mu, bl_sigma = bl(cap_w, sigma_prior, P, Q, Omega, delta, tau)

    if portfolio == 'msr':
        w = maximize_sharpe(rf, bl_mu, bl_sigma)
    elif portfolio == 'gmv':
        w = lowest_volatility(bl_sigma)
    elif portfolio == 'erc':
        w = equal_risk_contributions(bl_sigma)
    else:
        raise ValueError("portfolio must be 'msr', 'gmv' or 'erc'")

    return w

# ENC parameter

def enc(weights):

    return ((weights**2).sum())**-1

# ENCB parameter

def encb(r, weights):
    """
    Effective Number of Correlated Bets.
    """
    cov = r[weights.index].cov()
    
    rc = risk_cont(weights, cov)
    
    return ((rc**2).sum())**-1

# Measure the similarity between covariances

def covar_similarity(A, B):
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape:
        return 0.0

    sign_A, logdet_A = np.linalg.slogdet(A)
    sign_B, logdet_B = np.linalg.slogdet(B)
    mid = (A + B) / 2.0
    sign_mid, logdet_mid = np.linalg.slogdet(mid)

    if sign_A <= 0 or sign_B <= 0 or sign_mid <= 0:
        raise ValueError("Matrices must be positive-definite.")

    # log(similarity) = 0.25*logdet_A + 0.25*logdet_B - 0.5*logdet_mid
    log_similarity = 0.25 * logdet_A + 0.25 * logdet_B - 0.5 * logdet_mid
    return float(np.exp(log_similarity))

# Delta calculated

def calculate_delta(returns, weights, rf=0.03):

    annual_r= annual_return(returns, 252)
    annual_v= annual_vol(returns, 252)

    covar= returns.cov()*252

    portf_r= portfolio_return(weights, annual_r)
    portf_v= portfolio_vol(weights, covar)

    return (portf_r - rf) / (portf_v**2)

# Number of components

def marchenko_pastur_bounds(n_assets, n_obs):

    # Use the marchenko pastur distribution to check the bounds of noisy data
    # Output lower bound and upper bound
    # The upper bound will be a threshold to consider a parameter as NO random

    q = n_obs / n_assets
    lambda_max = (1 + np.sqrt(1 / q)) ** 2
    lambda_min = (1 - np.sqrt(1 / q)) ** 2
    return lambda_min, lambda_max

# Real number of uncorrelated assets

def effective_number_of_bets(returns, weights=None):
    """
    Compute the effective number of independent bets (ENB) in a portfolio
    using PCA-based entropy, following Meucci's diversification framework.

    Intuition: if all the variance is explained by 1 component, you have
    ENB ~= 1 (one real bet, no matter how many assets you hold). If
    variance is spread evenly across N components, ENB ~= N (maximally
    diversified across your independent risk sources).

    returns : DataFrame of asset returns (rows = time, cols = assets)
    weights : optional portfolio weights (Series, indexed like returns.columns).
              If provided, computes ENB for YOUR portfolio's risk allocation
              specifically, not just the asset universe in general.
              If None, computes ENB for the asset universe as a whole
              (equivalent to assuming equal risk contribution).

    Returns
    -------
    dict with:
        'n_components'      : total number of components (= n_assets)
        'explained_variance' : variance ratio per component
        'enb'                : effective number of bets (float)
        'diversification_ratio' : enb / n_assets (0 to 1 scale — how close
                                    you are to full diversification across
                                    all possible independent sources)
    """
    n_assets = len(weights[weights>0].index)

    # Standardize returns (same reasoning as pca_covariance: puts each
    # asset on equal footing regardless of its raw volatility)
    X_scaled = StandardScaler().fit_transform(returns.values)

    # Fit PCA with ALL components (we need the full eigenvalue spectrum,
    # not just the top few, to measure concentration)
    pca = PCA(n_components=n_assets)
    pca.fit(X_scaled)

    if weights is None:
        # Variance explained by each component, as a probability distribution
        p = pca.explained_variance_ratio_
    else:
        # Project the portfolio's risk allocation onto the components:
        # this tells you how much of YOUR portfolio's variance (not just
        # the universe's variance) comes from each principal component.
        w = weights.reindex(returns.columns).values
        # Portfolio variance contribution per component:
        # component loadings weighted by portfolio weights, squared,
        # scaled by each component's eigenvalue
        loadings = pca.components_ @ w            # (n_components,)
        comp_var = (loadings ** 2) * pca.explained_variance_
        p = comp_var / comp_var.sum()

    # Shannon entropy of the variance distribution across components.
    # Add a tiny epsilon to avoid log(0) for components with ~0 variance.
    entropy = -np.sum(p * np.log(p + 1e-12))

    # Effective number of bets = exponential of entropy.
    # (This is the standard "effective N" transform: converts entropy,
    # which is in log-units, back into a count you can compare to n_assets.)
    enb = np.exp(entropy)

    return {
        "n_components": n_assets,
        "explained_variance": p,
        "enb": enb,
        "diversification_ratio": enb / n_assets,
    }

# Clean small weights

def clean_weights(weights, min_weight=0.01):
    """
    Zero out weights below min_weight and renormalize the rest to sum to 1.
    """
    w = weights.copy()
    w[w < min_weight] = 0.0
    if w.sum() == 0:
        raise ValueError("All weights fell below min_weight — threshold too high")
    return w / w.sum()