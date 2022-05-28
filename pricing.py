import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()


def vega(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: initial_num_days to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    vega = S * si.norm.cdf(d1, 0.0, 1.0) * np.sqrt(T)

    return vega


def newton_vol_call(S, K, T, C, r, sigma):
    # S: spot price
    # K: strike price
    # T: initial_num_days to maturity
    # C: Call value
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega

        return abs(xnew)


def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - P) / vega

        return abs(xnew)


def euro_vanilla(S, K, T, r, sigma, option='call'):
    # S: spot price
    # K: strike price
    # T: initial_num_days to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    if sigma == 0:
        sigma = 0.01

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = None

    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    assert result is not None, "result never found"

    return result

# print(euro_vanilla(389.63, 390, 5/365, 0.0075, 0.2748))