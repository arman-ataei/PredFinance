"""
load_dataset.py

Provides the load_dataset function to fetch and standardize Bitcoin OHLCV data
from Yahoo Finance using the yfinance package.

All column names will be flattened (if necessary) and converted to lower-case.

Returns: pd.DataFrame with columns: 'open', 'high', 'low', 'close', 'adj_close', 'volume', indexed by date_time.

Follows functional programming principle: input parameters are NOT mutated.
"""

import pandas as pd
import yfinance as yf
from typing import Tuple
from datetime import time as datetime_time, date as datetime_date, datetime
import os 



from streamlit_app.utils import get_now

class FetchDataError(Exception):
    def __init__(self, message="Faild To Fetch Data."):
        super().__init__(message)



def load_dataset(
    ticker: str = "BTC-USD",
    period: str = "max",  # "max", "1y", "5y", "1mo", etc.
    interval: str = "1h", # "1d", "1h", "15m", etc.
    max_tries:int= 3 ,
) -> pd.DataFrame:
    """
    Loads historical OHLCV Bitcoin data from Yahoo Finance using yfinance,
    flattens multi-level columns if present, standardizes column names to
    lowercase, and returns a clean, immutable DataFrame.

    Parameters
    ----------
    ticker : str, default "BTC-USD"
        Ticker symbol for the instrument (Bitcoin in USD).
    period : str, default "max"
        Historical period to download.
    interval : str, default "1d"
        Data frequency.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with lowercase column names, indexed by date_time.

    Mathematical Notes
    ------------------
    The data will have the following time series columns:
    - open_t:      Opening price at time t
    - high_t:      Highest price within period t
    - low_t:       Lowest price within period t
    - close_t:     Closing price at time t
    - adj_close_t: Adjusted closing price (may differ due to splits/dividends on stocks; for BTC-USD, typically same as close)
    - volume_t: Total traded volume in period t
    """

    # Download the raw data
    df_raw = yf.download(ticker, period=period, interval=interval)
    i = 0
    while df_raw.empty and max_tries<i:
        df_raw = yf.download(ticker, period=period, interval=interval)

    if df_raw.empty:
        raise FetchDataError

    # If columns are multi-level (MultiIndex), flatten to single-level strings
    if isinstance(df_raw.columns, pd.MultiIndex):
        # Join levels with underscores (e.g., ("Adj Close", "") → "adj_close")
        df_raw.columns =[str(col[0]).lower() for col in df_raw.columns.values]
    else:
        # Convert existing column names to lowercase
        df_raw.columns = [col.lower() for col in df_raw.columns]

    # Reset index to ensure 'Datetime' is a column, then set as index again
    df_raw = df_raw.reset_index()
    # print(df_raw.columns)
    if 'Datetime' in df_raw.columns:
        df_raw['Datetime'] = pd.to_datetime(df_raw['Datetime'])
        df_raw.rename(columns={"Datetime": 'date_time'}, inplace=True)
        df_raw.set_index('date_time',inplace=True)

    elif "Date" in df_raw.columns:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw.rename(columns={"Date": 'date_time'}, inplace=True)
        df_raw.set_index('date_time',inplace=True)


    return df_raw.copy()  # return a new object to avoid side effects

def load_dataset_from_disk(ds_path:str, )->pd.DataFrame:
    '''
    assuming that the dataset contains `date_time` column
    '''
    try:
        df = pd.read_csv(ds_path, index_col='date_time')
        df = df.reset_index()
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)
        return df 

    except Exception as e:
        print(f"Something went Wrong while reading the file: {e}")
    

    

def extend_dataset(prev_df: pd.DataFrame,
                   latest_df:pd.DataFrame, 
                   dataset_path:str=os.path.join('.', 'src','dataset'))->Tuple[pd.DataFrame, str]:
    """
    This function extends the dataset with new data

    dataset_path: the path to the folder to save the output of the function
    """
     
   
    now = get_now()


    latest_df_path = os.path.join(dataset_path,'raw-btc-'+now)
    
    # converting the index to datetime 
    df = pd.concat([prev_df, latest_df])
    df = df.reset_index()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time',inplace=True)

    # removing duplicated indices
    df = df[~df.index.duplicated(keep='first')]

    # saving the combined dataset to the disk
    df.to_csv(f"{latest_df_path}.csv")
    print('Dataset updated successfully.')

    dataset_files_names = sorted(os.listdir(dataset_path))
    ds_files = []
    for name in dataset_files_names:
        if os.path.isfile(os.path.join(dataset_path, name)):
            ds_files.append(name)
    
    dataset_files_names = ds_files
    del ds_files

    if len(dataset_files_names)>1:
        dataset_files_names.sort()
        archive_path = os.path.join(dataset_path, 'archive')
        try:
            for ds in dataset_files_names[:-1]:
                os.rename(os.path.join(dataset_path,ds), os.path.join(archive_path,ds))
            
            print("Previous datasets moved to the archive.")
        except Exception as e:
            print(f"When moving files to archive something went wrong. {e}")

    return df, latest_df_path


def save_df(df:pd.DataFrame,
            time_frame: str,
            ds_path:str= os.path.join('.', 'src','dataset'))->str:
    
    now = get_now()
    
    ds_path_ = os.path.join(ds_path, time_frame)
    os.makedirs(ds_path_, exist_ok=True)
    
    latest_df_path = os.path.join(ds_path,'raw-btc-'+now)
    df.to_csv(f"{latest_df_path}.csv")
    
    return latest_df_path

def save_df_featured(df_featured: pd.DataFrame, 
                     time_frame: str,
                     ds_path:str= os.path.join('.', 'src','dataset'))->Tuple[pd.DataFrame, str]:
    
    ds_featured_path = os.path.join(ds_path, 'featured', time_frame)
    os.makedirs(ds_featured_path, exist_ok=True)
    archive_path = os.path.join(ds_featured_path, 'archive')
    os.makedirs(archive_path, exist_ok=True)

    ds_featured_list = sorted(os.listdir(ds_featured_path))
    ds_featured_csv = []
    for name in ds_featured_list:
        if os.path.isfile(os.path.join(ds_featured_path, name)):
            ds_featured_csv.append(name)
    del ds_featured_list

    if len(ds_featured_csv):
        archive_path = os.path.join(ds_featured_path, 'archive')
        try:
            for ds in ds_featured_csv:
                os.rename(os.path.join(ds_featured_path,ds), os.path.join(archive_path,ds))
            
            print("Previous featured_datasets moved to the archive.")
        except Exception as e:
            print(f"While moving previous featured_datasets to archive, something went wrong. {e}")
    now = get_now()
    latest_df_featured_path = os.path.join(ds_featured_path,'featured-btc-'+now)
    df_featured.to_csv(os.path.join(ds_featured_path, latest_df_featured_path+'.csv'))
    print("New featured_dataset has been saved to the disk successfully.")
    return df_featured, os.path.join(ds_featured_path, latest_df_featured_path+'.csv')

# def get_now()->str:
#     now_time = str(datetime.now()).split()[1][:5] # removing mili-sec from the time
#     now_date = str(str(datetime.now()).split()[0]) # extracting current date 
#     now = "_".join([now_date,now_time]) #  now date-time

#     return now