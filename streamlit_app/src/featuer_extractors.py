"""
feature_extractor.py

Feature extraction for Bitcoin price prediction.

Implements immutable functional feature extraction routines, grouped by type.
All mathematical definitions/formulas and references are documented inline.

Each function:
- Accepts input DataFrame (OHLCV)
- Returns new DataFrame with added feature columns (does not mutate input)
- Handles missing values (NaNs) gracefully


"""

import numpy as np
import pandas as pd
import pywt
from typing import List
# -------------------------------
# 1. Price-Based Statistical Features
# -------------------------------

def add_log_returns(df: pd.DataFrame, 
                    close: str = 'close', 
                    open:str='open',
                    high:str='high',
                    low:str='low') -> pd.DataFrame:
    """
    Adds logarithmic returns as a new column 'log_return'.

    Mathematical Definition:
        log_return_t = log(P_t / P_{t-1})

    Where:
        P_t   = price at time t (e.g., closing price)
        log_return_t = logarithmic return at time t

    .. math::
        \\text{log_return_open_to_close}_t = \\ln\\left(\\frac{\\text{Close}_t}{\\text{Open}_t}\\right)

    Returns a new DataFrame with 'log_return_close_to_close' and 'log_return_open_to_close' columns.
    """
    df2 = df.copy()
    assert close in df.columns
    assert open in df.columns
    assert high in df.columns
    assert low in df.columns
    
    df2["log_return_close_to_close"] = np.log(df2[close] / df2[close].shift(1))
    df2['log_return_open_to_close'] = np.log(df2[close] / df2[open])
    df2['log_return_high_low'] = np.log(df2[high] / df2[low])
    df2['log_return_open_prevclose'] = np.log(df2[open] / df2[close].shift(1))
    return df2

def add_volatilities(df: pd.DataFrame, 
                     return_cols: List[str] = ['log_return_close_to_close', 'log_return_open_to_close', 'log_return_high_low', 'log_return_open_prevclose' ], 
                     window: int = 20) -> pd.DataFrame:
    """
    Adds rolling volatility (standard deviation of log returns) over a given window.

    Mathematical Definition:
        volatility_t = stddev( log_return_{t-window+1}, ..., log_return_t )

    Where stddev(x) = sqrt(1/n * sum_i (x_i - mean(x))^2)

    .. math::
        \\text{vol}_t = \\sqrt{\\frac{1}{N-1}\\sum_{i=0}^{N-1} (r_{t-i} - \\bar{r})^2}
    
    """
    df2 = df.copy()
    
    for col in return_cols:
        assert col in df.columns

    if np.sum(np.array([ret_col not in df2.columns for ret_col in return_cols])):
        df2 = add_log_returns(df2)
    
    for return_col in return_cols:
        df2[f"volatility_{return_col}_{window}"] = df2[return_col].rolling(window=window).std()
    return df2


# -------------------------------
# 2. Momentum Indicators
# -------------------------------

def add_rsi(df: pd.DataFrame, 
            col: str = 'close', 
            window: int = 14) -> pd.DataFrame:
    """
    Adds Relative Strength Index (RSI) feature.

    Mathematical Definition:
        RSI_t = 100 - 100 / (1 + RS_t)
        RS_t = mean(Gains_t) / mean(Losses_t)
        Gains_t = [ max(ΔP_i, 0) for i in window ]
        Losses_t = [ abs(min(ΔP_i, 0)) for i in window ]
        ΔP_i = P_i - P_{i-1}
    """
    df2 = df.copy()
    delta = df2[col].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df2.index).rolling(window=window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=df2.index).rolling(window=window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))


    df2[f"rsi_{window}"] = rsi

    return df2

def add_macd(df: pd.DataFrame, 
             col: str = 'close', 
             span_fast: int = 12, 
             span_slow: int = 26, 
             span_signal: int = 9) -> pd.DataFrame:
    """
    Adds MACD (Moving Average Convergence Divergence) and its signal, histogram.

    Mathematical Definitions:
        EMA_{t, α}(x) = α*x_t + (1-α)*EMA_{t-1, α}(x)
        MACD_t = EMA_{fast}(P_t) - EMA_{slow}(P_t)
        Signal_t = EMA_{signal}(MACD_t)
        Histogram_t = MACD_t - Signal_t

    Typical parameters: fast=12, slow=26, signal=9
    """
    df2 = df.copy()
    ema_fast = df2[col].ewm(span=span_fast, min_periods=span_fast, adjust=False).mean()
    ema_slow = df2[col].ewm(span=span_slow, min_periods=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=span_signal, min_periods=span_signal, adjust=False).mean()
    histogram = macd - signal

    df2["macd"] = macd
    df2["macd_signal"] = signal
    df2["macd_histogram"] = histogram
    return df2

def add_obv(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """
    Adds On-Balance Volume (OBV) indicator.

    Mathematical Definition:
        OBV_t = OBV_{t-1} + {
            +volume_t if P_t > P_{t-1}
            -volume_t if P_t < P_{t-1}
             0 if P_t == P_{t-1}
        }
        OBV_0 = 0

    OBV measures buying/selling pressure as cumulative volume adjusted for price moves.
    """
    df2 = df.copy()
    direction = np.sign(df2[price_col].diff().fillna(0))
    obv = (direction * df2[volume_col]).cumsum()
    df2["obv"] = obv
    return df2

# -------------------------------
# 3. Trend and Volatility Bands
# -------------------------------

def add_bollinger_bands(df: pd.DataFrame, col: str = 'close', window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Adds Bollinger Bands (upper, lower, middle).

    Mathematical Definitions:
        SMA_t = mean(P_{t-N+1}, ..., P_t)              (Simple Moving Average)
        STD_t = std(P_{t-N+1}, ..., P_t)               (Rolling standard deviation)
        Upper_t = SMA_t + k * STD_t
        Lower_t = SMA_t - k * STD_t

    Where: N=window, k=num_std (usually 2)
    """
    df2 = df.copy()
    sma = df2[col].rolling(window=window).mean()
    std = df2[col].rolling(window=window).std(ddof=0)
    df2["bollinger_mid"] = sma
    df2["bollinger_upper"] = sma + num_std * std
    df2["bollinger_lower"] = sma - num_std * std
    return df2

def add_moving_averages(df: pd.DataFrame, col: str = 'close', windows: list = [20, 50, 200]) -> pd.DataFrame:
    """
    Adds multiple Simple Moving Averages (SMAs).

    Mathematical Definition:
        SMA_t^N = mean(P_{t-N+1}, ..., P_t)
    """
    df2 = df.copy()
    for w in windows:
        df2[f"sma_{w}"] = df2[col].rolling(window=w).mean()
    return df2

# -------------------------------
# 4. Spectral and Complexity Features
# -------------------------------

def add_wavelet_energy(df: pd.DataFrame, col: str = 'close', wavelet: str = 'morl', scales: np.ndarray = None) -> pd.DataFrame:
    """
    Adds Continuous Wavelet Transform (CWT) spectral energy as a new feature.

    Mathematical Definition:
        CWT(s, t) = ∫ x(τ) * ψ^*_{s, t}(τ) dτ
        Energy_t = ∑_{scales} |CWT(scale, t)|^2

    - x(t): the time series (e.g., close price)
    - ψ: mother wavelet
    - scales: array of scales
    """
    df2 = df.copy()
    # Default: dyadic scales
    if scales is None:
        scales = np.logspace(1, 3, num=32)  # e.g., scales 10 to 1000

    coeffs, _ = pywt.cwt(df2[col].fillna(method='ffill').values, scales, wavelet)
    energy = np.sqrt(np.sum(np.abs(coeffs) ** 2, axis=0))  # L2 norm across scales for each t
    df2["wavelet_energy"] = pd.Series(energy, index=df2.index)
    return df2

# --- Example: Compose all feature groups in a composable, non-mutating way ---
def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compose all feature extractors, returning a new DataFrame with all features.
    """
    out = add_log_returns(df)
    out = add_volatilities(out)
    # TODO
    # out = add_autocorrelations(out)  
    out = add_rsi(out)
    out = add_macd(out)
    out = add_obv(out)
    out = add_bollinger_bands(out)
    out = add_moving_averages(out)
    out = add_wavelet_energy(out)
    return out
