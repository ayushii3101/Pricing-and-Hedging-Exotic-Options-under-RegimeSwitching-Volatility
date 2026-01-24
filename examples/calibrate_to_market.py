"""
Market Calibration Script
=========================

Fits the Regime-Switching Heston model to historical data (e.g., S&P 500).
Requires: yfinance, hmmlearn
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Calibration")

def fetch_data(ticker="SPY", start="2015-01-01", end="2023-01-01"):
    """Fetch historical data with robust column handling."""
    logger.info(f"Fetching data for {ticker}...")
    
    # Download data
    # auto_adjust=False attempts to force 'Adj Close' to appear.
    data = yf.download(ticker, start=start, end=end, progress=True, auto_adjust=False)
    
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}. Check internet or ticker symbol.")
    
    # FIX: Handle MultiIndex columns (Common in newer yfinance versions)
    # If columns look like ('Adj Close', 'SPY'), flatten them.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # FIX: Fallback if 'Adj Close' is missing (use 'Close' instead)
    if 'Adj Close' in data.columns:
        price_col = 'Adj Close'
    elif 'Close' in data.columns:
        logger.warning("'Adj Close' not found. Using 'Close' price.")
        price_col = 'Close'
    else:
        raise KeyError(f"Could not find price column. Available columns: {data.columns}")
    
    # Calculate returns and volatility
    data['LogReturn'] = np.log(data[price_col] / data[price_col].shift(1))
    data['RealizedVol'] = data['LogReturn'].rolling(window=21).std() * np.sqrt(252)
    
    return data.dropna()

def fit_regimes(data, n_states=3):
    """Fit Gaussian HMM to realized volatility."""
    logger.info(f"Fitting {n_states}-state HMM to market data...")
    
    # We fit the HMM on Realized Volatility to identify "Vol Regimes"
    X = data['RealizedVol'].values.reshape(-1, 1)
    
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    
    # Sort regimes by volatility (Low, Med, High)
    means = model.means_.flatten()
    sorted_idx = np.argsort(means)
    
    logger.info("Calibration Complete. Identified Regimes:")
    
    params = []
    names = ['Low Vol', 'Medium Vol', 'High Vol']
    
    for i, idx in enumerate(sorted_idx):
        vol = means[idx]
        var = model.covars_[idx][0][0]
        regime_name = names[i] if i < 3 else f"Regime {i}"
        
        logger.info(f"  {regime_name}: Vol={vol:.2%}, Var_of_Vol={var:.2e}")
        
        # Estimate stylized parameters based on regime
        # (In a full prod system, we'd MLE these too, but this fits the HMM logic)
        params.append({
            'name': regime_name,
            'volatility': float(vol),
            'drift': 0.08 if i == 0 else (0.04 if i == 1 else -0.10)
        })
        
    return params, model.transmat_[sorted_idx][:, sorted_idx]

def main():
    try:
        # 1. Get Data
        data = fetch_data()
        
        # 2. Fit HMM
        regime_params, trans_matrix = fit_regimes(data)
        
        # 3. Print Output
        print("\n" + "="*50)
        print("RECOMMENDED CONFIGURATION (Copy to model_config.yaml)")
        print("="*50)
        
        print("regimes:")
        for i, p in enumerate(regime_params):
            print(f"  regime_{i+1}:")
            print(f"    name: \"{p['name']}\"")
            print(f"    drift: {p['drift']}")
            print(f"    volatility: {p['volatility']:.4f}")
            print(f"    volatility_type: \"heston\"")
            print(f"    # Calibrated from HMM")
            
        print("\ntransition_matrix:")
        for row in trans_matrix:
            print(f"  - {row.tolist()}")
            
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        logger.info("Ensure 'yfinance' and 'hmmlearn' are installed.")

if __name__ == "__main__":
    print("DEBUG: Script started...")
    main()