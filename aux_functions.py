import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime,timedelta


def clean_tck_names(listatck):
  cleantcks = [tck.split('.')[0] for tck in listatck]
  return cleantcks

def define_analysis_period(df, kyears, kmonths):
  df.index = pd.to_datetime(df.index)
  last_date = df.index[-1]
  beginningdate = last_date - pd.DateOffset(years= kyears, months = kmonths)
  df = df[df.index > beginningdate]
  return df

def calculate_daily_returns(df):
  df1 = np.log(df).diff().iloc[1:]
  return df1

def add_sector_constraint(df,x, sector, sign, weight):
    t1 = np.where(df['label']== sector, 1, 0)

    if sign == '=':
        cons = cp.sum(cp.multiply(x,t1))== weight
    elif sign == '>=':                                                          
        cons = cp.sum(cp.multiply(x,t1))>= weight
    elif sign == '<=':
        cons = cp.sum(cp.multiply(x,t1))>= weight
    return cons       

def obtain_max_port(returns, w, constraints):
    ren = np.array(returns.mean())
    objective = cp.sum(cp.multiply(w, ren))
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    seleccion_df=pd.DataFrame(w.value.round(3), index=returns.columns, columns= ['Peso'])
    seleccion_df = seleccion_df.loc[seleccion_df['Peso'] != 0] 
    seleccion_df['company']=seleccion_df.index
    return seleccion_df
def obtain_min_vol_port(returns, w, constraints):
    
    sigma = returns.cov().values
    port_risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Minimize(port_risk), constraints)
    prob.solve()
    seleccion_df=pd.DataFrame(w.value.round(3), index=returns.columns, columns= ['Peso'])
    seleccion_df = seleccion_df.loc[seleccion_df['Peso'] != 0] 
    seleccion_df['company']=seleccion_df.index
    return seleccion_df

def obtain_optimal_port(ret_data,risk_data, portfolio_weights, returns):
    sharpes = ret_data/risk_data 
    idx = np.argmax(sharpes)
    optimal_ret, optimal_risk = ret_data[idx], risk_data[idx]
    seleccion_df = pd.DataFrame(portfolio_weights[idx],index=returns.columns, columns= ['Peso']).round(3)
    seleccion_df = seleccion_df.loc[seleccion_df['Peso'] != 0] 
    seleccion_df['company']=seleccion_df.index
    return seleccion_df
    


def efficient_frontier(returns, constraints):
    """
    construye un conjunto de problemas de programación cuádrática
    para inferir la frontera eficiente de Markovitz. 
    En cada problema el parámetro gamma se cambia para aumentar
    la penalización del riesgo en la función de maximización.
    """
    n_samples=50
    gamma_low=-1
    gamma_high=10
    sigma = returns.cov().values
    mu = np.mean(returns, axis=0).values  
    n = sigma.shape[0]        
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, sigma)
    
    prob = cp.Problem(cp.Maximize(ret - gamma*risk), [cp.sum(w) == 1, w >= 0]) 
    # Equivalente 
    #prob = cp.Problem(cp.Minimize(risk - gamma*ret), 
    #                  [cp.sum(w) == 1,  w >= 0])   
    risk_data = np.zeros(n_samples)
    ret_data = np.zeros(n_samples)
    gamma_vals = np.logspace(gamma_low, gamma_high, num=n_samples)
    
    portfolio_weights = []    
    for i in range(n_samples):
        gamma.value = gamma_vals[i]
        prob.solve()
        risk_data[i] = np.sqrt(risk.value)
        ret_data[i] = ret.value
        portfolio_weights.append(w.value)   
    return ret_data, risk_data, gamma_vals, portfolio_weights