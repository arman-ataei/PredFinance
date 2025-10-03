
from pandas import DataFrame
from datetime import time as datetime_time, date as datetime_date, datetime

def get_now()->str:
    now_time = str(datetime.now()).split()[1][:5] # removing mili-sec from the time
    now_date = str(str(datetime.now()).split()[0]) # extracting current date 
    now = "_".join([now_date,now_time]) #  now date-time

    return now


def resample_ohlcv(df:DataFrame, timeframe:str="2H"):
    """
    Resample OHLCV dataframe to a new timeframe.
    df: DataFrame with columns ['open','high','low','close','volume'] and DatetimeIndex
    timeframe: e.g. '15T', '1H', '4H', '1D'
    """

    df2 = df.copy()
    df2 = df.resample(timeframe).agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum"
                    })

    return df2.dropna()