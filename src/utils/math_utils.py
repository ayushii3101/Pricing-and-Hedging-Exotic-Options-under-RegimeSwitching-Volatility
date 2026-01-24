"""
Mathematical Utilities
=====================

Common mathematical functions for quantitative finance.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Black-Scholes formula for European call option.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float
        Dividend yield
        
    Returns
    -------
    float
        Call option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Black-Scholes formula for European put option.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float
        Dividend yield
        
    Returns
    -------
    float
        Put option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Calculate Black-Scholes Delta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes Gamma."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes Vega."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Calculate Black-Scholes Theta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        return term1 + term2
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return term1 + term2


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Calculate Black-Scholes Rho."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    max_iter: int = 100,
    tol: float = 1e-6
) -> Optional[float]:
    """
    Compute implied volatility using Newton-Raphson method.
    
    Parameters
    ----------
    price : float
        Observed option price
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    Optional[float]
        Implied volatility, or None if convergence fails
    """
    sigma = 0.3  # Initial guess
    
    for i in range(max_iter):
        if option_type == 'call':
            price_estimate = black_scholes_call(S, K, T, r, sigma)
        else:
            price_estimate = black_scholes_put(S, K, T, r, sigma)
        
        vega = bs_vega(S, K, T, r, sigma)
        
        if vega < 1e-10:
            return None
        
        diff = price - price_estimate
        
        if abs(diff) < tol:
            return sigma
        
        sigma += diff / vega
        
        if sigma <= 0:
            return None
    
    return None


def normal_cdf_inverse(p: float) -> float:
    """Inverse of standard normal CDF."""
    return norm.ppf(p)


def correlation_matrix(rhos: np.ndarray) -> np.ndarray:
    """
    Generate correlation matrix from correlations.
    
    Parameters
    ----------
    rhos : np.ndarray
        Array of correlations
        
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    n = len(rhos)
    corr = np.eye(n + 1)
    
    for i in range(n):
        corr[i, i + 1] = rhos[i]
        corr[i + 1, i] = rhos[i]
    
    return corr


def cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition of correlation matrix.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix
        
    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor
    """
    return np.linalg.cholesky(corr_matrix)


def generate_correlated_normals(
    n_samples: int,
    correlations: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate correlated normal random variables.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    correlations : np.ndarray
        Correlation matrix
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Correlated normal samples, shape (n_samples, n_vars)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_vars = correlations.shape[0]
    L = cholesky_decomposition(correlations)
    
    # Generate independent normals
    Z = np.random.randn(n_samples, n_vars)
    
    # Transform to correlated
    X = Z @ L.T
    
    return X


def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate log returns from price series.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
        
    Returns
    -------
    np.ndarray
        Log returns
    """
    return np.diff(np.log(prices))


def realized_volatility(returns: np.ndarray, annualization: float = 252) -> float:
    """
    Calculate realized volatility from returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    annualization : float
        Annualization factor (252 for daily, 12 for monthly)
        
    Returns
    -------
    float
        Realized volatility
    """
    return np.std(returns) * np.sqrt(annualization)
