"""
visualization.py

Visualization utilities for Bitcoin price prediction features.

Each function:
- Accepts a DataFrame with extracted features (from feature_extractor.py)
- Returns a Plotly Figure object
- Does not mutate the input DataFrame
- Includes inline comments with mathematical definitions and references

References:
- Tsay, R. S. (2010). Analysis of Financial Time Series. Wiley.
- Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
- Appel, G. (2005). Technical Analysis: Power Tools for Active Investors.
- Granville, J. E. (1963). Granville's New Key to Stock Market Profits.
- Bollinger, J. (2001). Bollinger on Bollinger Bands.
- Murphy, J. J. (1999). Technical Analysis of the Financial Markets.
- Percival, D. B., & Walden, A. T. (2000). Wavelet Methods for Time Series Analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# -------------------------------
# 0. Single Columns
# -------------------------------
def plot_col(df: pd.DataFrame, col: str):
    fig_line = px.line(df, y=col, title=f"{col} over time")
    fig_hist = px.histogram(df, x=col, nbins=50, title=f"Distribution of {col}")
    return fig_line, fig_hist



# -------------------------------
# 1. Log Returns
# -------------------------------

def plot_log_returns(df: pd.DataFrame, col: str = 'log_return_close_to_close'):
    """
    Plot log returns as time series and histogram.

    Mathematical Definition:
        r_t = log(C_t / C_{t-1})   [Close-to-Close]
        r_t = log(C_t / O_t)       [Open-to-Close]

    Interpretation:
        - Large positive values → bullish interval
        - Large negative values → bearish interval
        - Histogram tails → risk of extreme moves
    """
    fig_line = px.line(df, y=col, title=f"{col} over time")
    fig_hist = px.histogram(df, x=col, nbins=50, title=f"Distribution of {col}")
    return fig_line, fig_hist

# -------------------------------
# 2. Volatility
# -------------------------------

def plot_volatility(df: pd.DataFrame, col: str = 'volatility_log_return_close_to_close_20'):
    """
    Plot rolling volatility.

    Mathematical Definition:
        σ_t = sqrt( (1/(N-1)) * Σ (r_{t-i} - mean)^2 )

    Interpretation:
        - Volatility clustering: calm vs turbulent periods
        - High volatility → reduce position size
    """
    return px.line(df, y=col, title=f"Rolling Volatility: {col}")

# -------------------------------
# 3. RSI
# -------------------------------

def plot_rsi(df: pd.DataFrame, col: str = 'rsi_14'):
    """
    Plot RSI indicator.

    Mathematical Definition:
        RSI_t = 100 - 100 / (1 + RS_t)
        RS_t = EMA(gains) / EMA(losses)

    Interpretation:
        - RSI > 70 → overbought (potential sell)
        - RSI < 30 → oversold (potential buy)
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI Indicator", yaxis=dict(range=[0,100]))
    return fig

# -------------------------------
# 4. MACD
# -------------------------------

def plot_macd(df: pd.DataFrame):
    """
    Plot MACD, Signal, and Histogram.

    Mathematical Definition:
        MACD_t = EMA_fast(P_t) - EMA_slow(P_t)
        Signal_t = EMA_signal(MACD_t)
        Histogram_t = MACD_t - Signal_t

    Interpretation:
        - Bullish crossover (MACD > Signal) → buy
        - Bearish crossover (MACD < Signal) → sell
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], mode='lines', name='Signal'))
    fig.add_trace(go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram'))
    fig.update_layout(title="MACD Indicator")
    return fig

# -------------------------------
# 5. OBV
# -------------------------------

def plot_obv(df: pd.DataFrame, price_col:str="close"):
    """
    Plot On-Balance Volume (OBV).

    Mathematical Definition:
        OBV_t = OBV_{t-1} + {+V_t if C_t > C_{t-1}, -V_t if C_t < C_{t-1}}

    Interpretation:
        - Rising OBV with rising price → strong trend
        - Divergence → weak trend
    """
    fig = go.Figure()

    # Price trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name=price_col
    ))

    # OBV trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['obv'],
        mode='lines',
        name='OBV',
        yaxis='y2'
    ))

    # Layout with dual y-axes
    fig.update_layout(
        title="Price and On-Balance Volume (OBV)",
        xaxis_title="Date",
        yaxis=dict(title=price_col),
        yaxis2=dict(title="OBV", overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99)
    )

    return fig

# -------------------------------
# 6. Bollinger Bands
# -------------------------------

def plot_bollinger(df: pd.DataFrame):
    """
    Plot Bollinger Bands with price.

    Mathematical Definition:
        Upper_t = SMA_t + k*σ_t
        Lower_t = SMA_t - k*σ_t

    Interpretation:
        - Price near upper band → overbought
        - Price near lower band → oversold
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_upper'], mode='lines', name='Upper'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_lower'], mode='lines', name='Lower'))
    fig.update_layout(title="Bollinger Bands")
    return fig

# -------------------------------
# 7. Moving Averages
# -------------------------------
from src.featuer_extractors import add_moving_averages
def plot_moving_averages(df: pd.DataFrame, windows=[20,50,200]):
    """
    Plot price with multiple SMAs.

    Mathematical Definition:
        SMA_t^N = (1/N) Σ P_{t-i}

    Interpretation:
        - Golden Cross (short SMA > long SMA) → buy
        - Death Cross (short SMA < long SMA) → sell
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close'))
    for w in windows:
        col = f"sma_{w}"
        if col not in df.columns:
            df = add_moving_averages(df, col='close', windows= [w])
        
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    
    fig.update_layout(title="Moving Averages")
    return fig

# -------------------------------
# 8. Wavelet Energy
# -------------------------------

def plot_wavelet_energy(df: pd.DataFrame):
    """
    Plot Wavelet Energy.

    Mathematical Definition:
        E_t = sqrt( Σ |CWT_j(t)|^2 )

    Interpretation:
        - Peaks in energy → regime shifts or volatility bursts
    """
    return px.line(df, y='wavelet_energy', title="Wavelet Energy")

# ---------------------------------
# 9. All figures in the same frame 
# ---------------------------------

def merge_figures(*figs):
    '''
    This function takes in all figures and plot them in
    a single frame
    '''
    merged_fig = go.Figure()
    for fig in figs:
        for trace in fig.data:
            merged_fig.add_trace(trace)
    merged_fig.update_layout(title="Combined Plot")
    return merged_fig
