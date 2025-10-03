"""
preprocessors.py

Preprocessing functions for time-series modeling:
- Train/Validation/Test split for time series
- Normalization (StandardScaler, MinMaxScaler)
- Polynomial feature expansion
- Dimensionality reduction (PCA, LDA)

All mathematical steps and transformation formulas given as comments.

Each function is stateless: input arrays are never mutated.
"""


import numpy as np
import pandas as pd
from typing import Tuple, Union

from streamlit_app import config
from streamlit_app.src.load_dataset import  get_now
    
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from typing import List, Dict, Literal, Union
import pickle
import os
# -----------------------------------------
# 1. Train/Validation/Test Splitting
# -----------------------------------------

def time_series_split(
    df: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    datetime_col: str = "date_time",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time series DataFrame into train/val/test partitions by temporal order.

    Steps:
        1. Optionally, sort by datetime_col if provided.
        2. Split sequentially: first train_size%, then val_size%, rest as test.

    Returns three DataFrames.

    Note: For true time series, random shuffling MUST be avoided. 
    """
    df2 = df.copy()
    df2.dropna(inplace=True)
    df2.reset_index(inplace=True)
    
    if datetime_col:
        df2 = df2.sort_values(datetime_col)
    
    df2[datetime_col] = pd.to_datetime(df2[datetime_col])
    df2.set_index(datetime_col, inplace=True)
    
    n = len(df2)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train = df2.iloc[:train_end].copy()
    val = df2.iloc[train_end:val_end].copy()
    test = df2.iloc[val_end:].copy()
    return train, val, test

# -----------------------------------------
# 2. Normalization & Scaling
# -----------------------------------------

def standard_scale(
    train: Union[pd.DataFrame, np.ndarray],
    val: Union[pd.DataFrame, np.ndarray],
    test: Union[pd.DataFrame, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardizes features via z-score scaling fit on train set only.

    Mathematical Formula:
        z = (x - μ) / σ

        Where:
           - μ = mean(train), σ = std(train)
           - For each feature independently

    Returns scaled arrays and the StandardScaler object (for inverse_transform). (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler

def minmax_scale(
    train: Union[pd.DataFrame, np.ndarray],
    val: Union[pd.DataFrame, np.ndarray],
    test: Union[pd.DataFrame, np.ndarray],
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scales features to a fixed range fit on train set only.

    Mathematical Formula:
        x_scaled = (x - min) / (max - min) * (range[1] - range[0]) + range[0]
        Where min, max = computed from train set only.

    Returns scaled arrays and the MinMaxScaler object.(train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler

# -----------------------------------------
# 3. Polynomial Feature Expansion
# -----------------------------------------

def poly_expand(
    X: Union[pd.DataFrame, np.ndarray],
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """
    Expands features by generating polynomial combinations up to specified degree.

    For input features x_1, x_2:
        degree=2 → [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2] (including bias)

    Returns expanded array and PolynomialFeatures object.

    Warning: Feature dimension increases polynomially with degree.(X_poly, poly)
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly, columns=feature_names, index=X.index), poly

# -----------------------------------------
# 4. Dimensionality Reduction
# -----------------------------------------

def pca_reduce(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: Union[int, float] = .9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Applies Principal Component Analysis (PCA) for linear dimensionality reduction.

    Steps:
        1. Fit PCA on X_train only, then transform all splits.
        2. PCA finds directions (principal components) maximizing variance.

    

    Returns dimension-reduced arrays and the fitted PCA object. (X_train_pca, X_val_pca, X_test_pca, pca)
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca



from .featuer_extractors import (add_log_returns,
                                 add_volatilities,
                                 add_moving_averages,
                                 add_macd,
                                 add_obv,
                                 add_rsi,
                                 add_bollinger_bands,
                                 add_wavelet_energy,
                                 extract_all_features)
class LoadAndPrepareData:
    def __init__(self, 
                 raw_data: pd.DataFrame,
                #  samples:List[int] = [1,2,3,4],
                interval:str,
                window_size:int = 5,
                LoadAndPrepareData_path:str = config.LOAD_AND_PREPARE_DIR,
                 ):
        """
        Process Dataset including: 
                - created time_framed_data   
                - adding all features to the dataset including:
                        - log-returns
                        - etc
        Inputs:
                - raw_data: the pandas dataframe fetched by `load_dataset`
                - interval: the value of `interval` passed to `load_dataset`
        
        Attributes: 
                - interval
                - raw_data
                - etc
        """
        self.window_size:int = window_size

        self.interval = interval
        self.raw_data = raw_data.copy()
        
        # creating the directory 
        self.main_dir = os.path.join(LoadAndPrepareData_path, self.interval)
        os.makedirs(self.main_dir, exist_ok=True)

        self.obj_path = os.path.join(self.main_dir, f'prp_{self.window_size}.obj')

        # self.__resample_ohlcv()
        self.featured_data:pd.DataFrame = raw_data.copy()
        
        self.poly_extractor = None
        self.poly_featured_data:pd.DataFrame = pd.DataFrame()

        
        self.reducer:PCA = None
        self.reduced_data:np.ndarray = np.array([])

        
        self.scaler:Union[MinMaxScaler, StandardScaler] = None
        self.scaled_data: Dict[str, np.ndarray]={'x_train':None,
                                                            'y_train':None,
                                                            'x_val':None,
                                                            'y_val':None,
                                                            'x_test':None,
                                                            'y_test':None} 
                        


        self.splited_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = \
                                                        {
                                                            "classification":
                                                                                {'x_train':None,
                                                                                'y_train':None,
                                                                                'x_val':None,
                                                                                'y_val':None,
                                                                                'x_test':None,
                                                                                'y_test':None},
                                                            "regression":
                                                                                {'x_train':None,
                                                                                    'y_train':None,
                                                                                    'x_val':None,
                                                                                    'y_val':None,
                                                                                    'x_test':None,
                                                                                    'y_test':None},
                                                        } 
        self.generators_cls= None
        self.generators_reg= None


        self.target_cls:  np.ndarray = np.array([]) 
        self.target_reg:  np.ndarray = np.array([]) 
       
        self.input_data:  np.ndarray = np.array([]) 
       

        self.input_data_dateTime:  np.ndarray = np.array([]) 
        # np.datetime_data


    def __resample_ohlcv(self,
                         periods: List[int] = [1,2,3,4]
                         )->pd.DataFrame:
        """
        Resample OHLCV dataframe to a new timeframe.
        df: DataFrame with columns ['open','high','low','close','volume'] and DatetimeIndex
        timeframe: e.g. '1H', '2H', '4H', etct
        """
       
        df2 = self.raw_data.copy()
        resamples = {
            period: pd.DataFrame()

            for period in periods
        }
        for period in periods:
            df2_ = df2.resample(period).agg({
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum"
                            }).copy()
            resamples[period] = df2_
            # df2 = load_dataset(interval=time_frame.lower())
            
        return resamples

    def add_log_returns(self,
                        close: str = 'close',
                        open: str = 'open',
                        high: str = 'high',
                        low: str = 'low')-> Dict[str, pd.DataFrame]:
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
        
        df_logs = add_log_returns(self.featured_data,close, open, high, low )
        self.featured_data = df_logs
        
        return self.featured_data.copy()
    
    def add_volatilities(self,
                        return_cols: List[str] = ['log_return_close_to_close', 'log_return_open_to_close', 'log_return_high_low', 'log_return_open_prevclose' ], 
                        window: int = 20)->Dict[str, pd.DataFrame]:
        """
        Adds rolling volatility (standard deviation of log returns) over a given window.

        Mathematical Definition:
            volatility_t = stddev( log_return_{t-window+1}, ..., log_return_t )

        Where stddev(x) = sqrt(1/n * sum_i (x_i - mean(x))^2)

        .. math::
            \\text{vol}_t = \\sqrt{\\frac{1}{N-1}\\sum_{i=0}^{N-1} (r_{t-i} - \\bar{r})^2}
        
        Returns A Dictionary of All time frames, each one contains the volatility features
        """
        
        for ret_col in return_cols:
            assert ret_col in self.featured_data.columns
        
        df_vol = add_volatilities(self.featured_data, return_cols, window)
        self.featured_data = df_vol
        
        return self.featured_data.copy()
    
    def add_rsi(self, 
                col: str = 'close', 
                window: int = 14) -> Dict[str, pd.DataFrame]:
        """
        Adds Relative Strength Index (RSI) feature.

        Mathematical Definition:
            RSI_t = 100 - 100 / (1 + RS_t)
            RS_t = mean(Gains_t) / mean(Losses_t)
            Gains_t = [ max(ΔP_i, 0) for i in window ]
            Losses_t = [ abs(min(ΔP_i, 0)) for i in window ]
            ΔP_i = P_i - P_{i-1}
        """

        df_rsi = add_rsi(self.featured_data, col, window)
        self.featured_data = df_rsi
        
        return self.featured_data.copy()
    
    def add_macd(self, 
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
        df_macd = add_macd(self.featured_data, col, span_fast, span_slow, span_signal)
        self.featured_data = df_macd
       
        return self.featured_data.copy()
    
    def add_obv(self, 
                price_col: str = 'close', 
                volume_col: str = 'volume') -> pd.DataFrame:
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
        df_obv = add_obv(self.featured_data, price_col, volume_col)
        self.featured_data = df_obv
        
        return self.featured_data.copy()
    
    def add_bollinger_bands(self, 
                            col: str = 'close', 
                            window: int = 20, 
                            num_std: float = 2) -> pd.DataFrame:
        """
        Adds Bollinger Bands (upper, lower, middle).

        Mathematical Definitions:
            SMA_t = mean(P_{t-N+1}, ..., P_t)              (Simple Moving Average)
            STD_t = std(P_{t-N+1}, ..., P_t)               (Rolling standard deviation)
            Upper_t = SMA_t + k * STD_t
            Lower_t = SMA_t - k * STD_t

        Where: N=window, k=num_std (usually 2)
        """
        df_bb = add_bollinger_bands(self.featured_data, col, window, num_std)              
        self.featured_data = df_bb
        
        return self.featured_data.copy()

    def add_moving_averages(self, 
                            col: str = 'close', 
                            windows: list = [20, 50, 200]) -> pd.DataFrame:
        """
        Adds multiple Simple Moving Averages (SMAs).

        Mathematical Definition:
            SMA_t^N = mean(P_{t-N+1}, ..., P_t)
        """
        
        df_ma = add_moving_averages(self.featured_data, col, windows)
        self.featured_data = df_ma
        return self.featured_data.copy()
    
    def add_wavelet_energy(self, 
                           col: str = 'close', 
                           wavelet: str = 'morl', 
                           scales: np.ndarray = None) -> pd.DataFrame:
        """
        Adds Continuous Wavelet Transform (CWT) spectral energy as a new feature.

        Mathematical Definition:
            CWT(s, t) = ∫ x(τ) * ψ^*_{s, t}(τ) dτ
            Energy_t = ∑_{scales} |CWT(scale, t)|^2

        - x(t): the time series (e.g., close price)
        - ψ: mother wavelet
        - scales: array of scales
        """
        
        df_we = add_wavelet_energy(self.featured_data, col, wavelet, scales)
        self.featured_data = df_we
        return df_we
    
    def extract_all_features(self) -> Dict[str, pd.DataFrame]:
        """
        Compose all feature extractors, returning a new DataFrame with all features.
        """
        df_featured = self.add_log_returns()
        df_featured = self.add_volatilities()
        df_featured = self.add_rsi()
        df_featured = self.add_macd()
        df_featured = self.add_obv()
        df_featured = self.add_bollinger_bands()
        df_featured = self.add_moving_averages()
        df_featured = self.add_wavelet_energy()
        
        
        return df_featured
    
    def add_poly_features(
            self,
            degree: int = 2,
            interaction_only: bool = False,
            include_bias: bool = False
            ) -> Tuple[Dict[str, pd.DataFrame], PolynomialFeatures]:
        """
        Expands features by generating polynomial combinations up to specified degree.

        For input features x_1, x_2:
            degree=2 → [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2] (including bias)

        Returns expanded array and PolynomialFeatures object.

        Warning: Feature dimension increases polynomially with degree.(X_poly, poly)
        """
        
        
        df = self.featured_data.copy()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.sort_values('date_time',inplace=True)
        dt_col = df['date_time'].copy()
        df = df.drop(['date_time'],axis=1)
        df_, poly_expander = poly_expand(df, degree, interaction_only, include_bias)
        df_['date_time'] = dt_col
        poly_df =  pd.DataFrame(df_)
        poly_df['date_time'] = pd.to_datetime(poly_df['date_time'])
        poly_df.set_index('date_time', inplace=True)
        self.poly_featured_data = poly_df
        self.poly_extractor = poly_expander
        
        
        return self.poly_featured_data.copy(), poly_expander
    
    def build_numpy_inputs_targets_data(self, 
                                        cls_threshold:float=0.0015)\
        ->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
            preparing numpy input_data, target_cls and target_reg to feed to the machine learning model
            
        """
        
        df_ = self.featured_data.copy()
        df_.dropna(inplace=True)
        df_['target_reg'] = df_["log_return_close_to_close"].shift(-1)
        df_.dropna(inplace=True)
        

        df_['target_cls']= pd.cut(df_['target_reg'],
                    bins=[-np.inf, -cls_threshold, cls_threshold, np.inf],
                    labels=[0,1,2]).astype(int)
        df_.dropna(inplace=True)
        
        self.target_reg = df_['target_reg'].copy().values
        self.target_cls = df_['target_cls'].copy().values
        self.input_data = df_.drop(['target_cls', 'target_reg'], axis=1).copy().values
        self.input_data_dateTime = df_.index
       

        inp_ = self.input_data.copy() 
        target_cls_ = self.target_cls.copy() 
        target_reg_ = self.target_reg.copy() 
            
        return inp_, target_cls_, target_reg_
   
    def build_windowed_data(self,
                            batch_size:int=None)->Tuple[Dict[str, TimeseriesGenerator], 
                                                        Dict[str, TimeseriesGenerator]]:
        """
        This function prepares the dataset to feed to learning models using the 
        TimeseriesGenerator from tensorflow.keras.preprocessing.sequence
        
        returns two dictionaries, one for classification and the other for the regression
        each dictionary contains generators for each timeframe
        
        this is how it can be used to prepare data for training:

        X_reg, y_reg = gen_reg[0]
        X_cls, y_cls = gen_cls[0]
        """
        window_size = self.window_size
        batch_size_ = batch_size if batch_size else self.input_data.shape[0]
        generator_cls_= TimeseriesGenerator(self.input_data, 
                                            self.target_cls,
                                            length= window_size,
                                            batch_size=batch_size_)
        self.generators_cls = generator_cls_
        generator_reg_ =  TimeseriesGenerator(self.input_data, 
                                            self.target_reg,
                                            length= window_size,
                                            batch_size=batch_size_)
        
        self.generators_reg = generator_reg_


        return self.generators_cls, self.generators_reg

    def split_data(self,
                   train_size: float = 0.7,
                   val_size: float = 0.15,
                   ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Splits time series DataFrame into train/val/test partitions by temporal order.

        Steps:
            1. Optionally, sort by datetime_col if provided.
            2. Split sequentially: first train_size%, then val_size%, rest as test.

        Returns three DataFrames.

        Note: For true time series, random shuffling MUST be avoided. 
        """
        assert val_size+train_size < 1
        returned_obj= {
                        "classification":
                                            {'x_train':None,
                                            'y_train':None,
                                            'x_val':None,
                                            'y_val':None,
                                            'x_test':None,
                                            'y_test':None},
                        "regression":
                                            {'x_train':None,
                                                'y_train':None,
                                                'x_val':None,
                                                'y_val':None,
                                                'x_test':None,
                                                'y_test':None},
                        } 
    
        generator_cls_ = self.generators_cls
        generator_reg_ = self.generators_reg
        x_reg, y_reg = generator_reg_[0]
        x_cls, y_cls = generator_cls_[0]

        x_reg = x_reg.reshape(x_reg.shape[0], -1)
        x_cls = x_cls.reshape(x_cls.shape[0], -1)
        x_train_reg, x__reg, y_train_reg, y__reg = train_test_split(x_reg, 
                                                                    y_reg, 
                                                                    test_size=(1-train_size), 
                                                                    shuffle=False)
        x_val_reg, x_test_reg, y_val_reg, y_test_reg = train_test_split(x__reg, 
                                                                        y__reg, 
                                                                        test_size=(1-train_size-val_size), 
                                                                        shuffle=False)

        x_train_cls, x__cls, y_train_cls, y__cls = train_test_split(x_cls, 
                                                                    y_cls, 
                                                                    test_size=(1-train_size), 
                                                                    shuffle=False)
        x_val_cls, x_test_cls, y_val_cls, y_test_cls = train_test_split(x__cls, 
                                                                        y__cls, 
                                                                        test_size=(1-train_size-val_size), 
                                                                        shuffle=False)
        self.splited_data ={
            "classification" :{
                                'x_train':x_train_cls.copy(),
                                'y_train':y_train_cls.copy(),
                                'x_val':x_val_cls.copy(),
                                'y_val':y_val_cls.copy(),
                                'x_test':x_test_cls.copy(),
                                'y_test':y_test_cls.copy(),
                },

            "regression":{
                            'x_train':x_train_reg.copy(),
                            'y_train':y_train_reg.copy(),
                            'x_val':x_val_reg.copy(),
                            'y_val':y_val_reg.copy(),
                            'x_test':x_test_reg.copy(),
                            'y_test':y_test_reg.copy(),
                }

        }

        returned_obj = {
            "classification" :{
                                    'x_train':x_train_cls.copy(),
                                    'y_train':y_train_cls.copy(),
                                    'x_val':x_val_cls.copy(),
                                    'y_val':y_val_cls.copy(),
                                    'x_test':x_test_cls.copy(),
                                    'y_test':y_test_cls.copy(),
                    },

                "regression":{
                                'x_train':x_train_reg.copy(),
                                'y_train':y_train_reg.copy(),
                                'x_val':x_val_reg.copy(),
                                'y_val':y_val_reg.copy(),
                                'x_test':x_test_reg.copy(),
                                'y_test':y_test_reg.copy(),
                }
        }

        return returned_obj
    
    def reduce_dim(self, 
                   n_components: Union[int, float] = .99
                   )->Tuple[np.ndarray, np.ndarray, np.ndarray, PCA ]:
        pass

    def save(self):
        # archive_path = os.path.join(LoadAndPrepareData, 'archive')
        # os.makedirs(archive_path, exist_ok=True)

        # obj_path = os.path.join(self.main_dir, 'prp.obj')
        
        try :
            with open(self.obj_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Something went wrong when trying to dump the oject, {e}")
    
    @staticmethod
    def load(LoadAndPrepareData_path:str, window_size:int=5, interval:str="5m"):
        file_path = os.path.join(LoadAndPrepareData_path, interval, f'prp_{window_size}.obj')
        
        try:
            
            with open(file_path, 'rb') as f:
                loaded_ = pickle.load(f)
                return loaded_
        except Exception as e:
            print(f"Something went wrong when trying to dump the oject, {e}")