# streamlit_app.py
import os
import config
import pandas as pd
import streamlit as st
from typing import List
import os ,sys
st.title("📊 Data Visualizations")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from utils import get_data, update_states

from src.load_dataset import (save_df,
                              load_dataset, 
                              extend_dataset, 
                              save_df_featured,
                              load_dataset_from_disk, 
                              )
from src.featuer_extractors import extract_all_features

from src.preprocessing import LoadAndPrepareData

from src.visualizations_plotly import (
    plot_log_returns, 
    plot_volatility, 
    plot_rsi, 
    plot_macd, 
    plot_obv, 
    plot_bollinger, 
    plot_moving_averages, 
    plot_wavelet_energy
)
from dataclasses import dataclass, asdict, field

@dataclass
class States:
    data_raw_is_available:bool = False
    data_featured_is_available:bool = False
    data_is_hidden:bool = True
    prp: LoadAndPrepareData = field(default_factory=LoadAndPrepareData)

class DataAnalytics:
    def __init__(self):
        if 'states' not in st.session_state:
            prp_instance = LoadAndPrepareData(raw_data=pd.DataFrame(), interval='2m')
            st.session_state.states =  States(prp=prp_instance)


    def build_ui(self):

        


        _states = st.session_state.states

        st.sidebar.header("Settings")

         # Sidebar: initializing/positioning UI elements/widgets
        sidebar_timeFrame_selectBox = st.sidebar.selectbox(label="Time Frame", options=['2m'])
        sidebar_windowSize_selectBox = st.sidebar.selectbox(label="Window Size", options=['7'])
        sidebar_fetchPrevData_btn = st.sidebar.button('Fetch Prev Data')
        sidebar_fetchNewData_btn = st.sidebar.button('Fetch New Data')     
        sidebar_show_raw_data_cbx = st.sidebar.checkbox("Show Raw-Data", disabled=not _states.data_featured_is_available)
        sidebar_show_featured_data_cbx = st.sidebar.checkbox("Show Featured-Data", disabled=not _states.data_raw_is_available)
        sidebar_plot_log_returns_cbx = st.sidebar.checkbox('Plot Log Returns ', disabled= not _states.data_featured_is_available )
        sidebar_plot_volatilities_cbx = st.sidebar.checkbox('Plot Volatilities ', disabled= not _states.data_featured_is_available )
        sidebar_plot_rsi_cbx = st.sidebar.checkbox('Plot RSI ', disabled= not _states.data_featured_is_available )
        sidebar_plot_macd_cbx = st.sidebar.checkbox('Plot MACD ', disabled= not _states.data_featured_is_available )
        sidebar_plot_obv_cbx = st.sidebar.checkbox('Plot OBV ', disabled= not _states.data_featured_is_available )
        sidebar_plot_bollinger_cbx = st.sidebar.checkbox('Plot Bollinger Bands ', disabled= not _states.data_featured_is_available )
        sidebar_plot_movingAverages_cbx = st.sidebar.checkbox('Plot Moving Averages ', disabled= not _states.data_featured_is_available )
        
        
        


        # featching data
        

        
        
        if sidebar_fetchPrevData_btn and sidebar_timeFrame_selectBox and sidebar_windowSize_selectBox:
            
            prp = LoadAndPrepareData.load(LoadAndPrepareData_path=config.LOAD_AND_PREPARE_DIR,
                                        window_size=int(sidebar_windowSize_selectBox),
                                        interval=sidebar_timeFrame_selectBox)
            _states = States(**{**asdict(_states),
                                "data_raw_is_available":True,
                                "data_featured_is_available":True,
                                "prp":prp,
                                })
            self.__update_states(_states)
            _states = st.session_state.states
            st.rerun()
            

    
        
        if sidebar_fetchNewData_btn and sidebar_timeFrame_selectBox and sidebar_windowSize_selectBox:
            # print("=====================================sidebar_timeFRame_selectBox")
            # print(sidebar_timeFrame_selectBox)
            df_new = self.__fetch_df_new(time_frame=sidebar_timeFrame_selectBox)
            # print(df_new.shape)
            prp = LoadAndPrepareData(raw_data=df_new, 
                                    interval=sidebar_timeFrame_selectBox, 
                                    window_size=int(sidebar_windowSize_selectBox), 
                                    LoadAndPrepareData_path=config.LOAD_AND_PREPARE_DIR)
            
            with st.spinner(text="Extracting features and indicators from raw data ..."):
                _ = prp.extract_all_features()
                prp.save()

            _states = States(**{**asdict(_states),
                                "data_raw_is_available":True,
                                "data_featured_is_available":True,
                                "prp":prp,
                                })
            self.__update_states(_states)
            _states = st.session_state.states
            st.rerun()
     

        
        df_featured = st.session_state.states.prp.featured_data

        # show raw data
        if sidebar_show_raw_data_cbx:
            self.__show_raw_data()
            _states = st.session_state.states

        
        
        # show featured data
        if sidebar_show_featured_data_cbx:
                st.header("Featured Data")
                st.write(df_featured)

        # plotting log returns
        if sidebar_plot_log_returns_cbx:
            self.__plot_log_returns(df_featured)
            
        # plotting volatilities
        if sidebar_plot_volatilities_cbx:
            self.__plot_volatilities(df_featured)

        # plotting rsi
        if sidebar_plot_rsi_cbx:
            self.__plot_rsi(df_featured)
        
        # plotting macd
        if sidebar_plot_macd_cbx:
            self.__plot_macd(df_featured)

         # plotting obv
        if sidebar_plot_obv_cbx:
            self.__plot_obv(df_featured)
        
        # plotting bollinger bands
        if sidebar_plot_bollinger_cbx:
            self.__plot_bollinger(df_featured)
        
        # plotting moving averages
        if sidebar_plot_movingAverages_cbx:
            self.__plot_movingAverages(df_featured)

    
    def __show_raw_data(self):
        _states = st.session_state.states

        st.header('Raw Data')

        if _states.data_is_hidden:
            _states = States(**{**asdict(_states),"data_is_hidden" : False})
            self.__update_states(_states)

            _states = st.session_state.states
            st.write(_states.prp.raw_data)
            
                    

        else:
            new_states = States(**{**asdict(_states),"data_is_hidden" : True})
            self.__update_states(new_states)
            _states = st.session_state.states
            st.rerun() # Rebuild the entire ui
    
    
    @staticmethod
    def __fetch_df_prev()->pd.DataFrame:
        df_prev = pd.DataFrame()
        try:
            dataset_path_files = os.listdir(config.DATASET_DIR)
            prev_df_name = sorted(filter(lambda file: file.endswith(".csv"), dataset_path_files))[-1]
            prev_df_path =  os.path.join(config.DATASET_DIR, prev_df_name)
            df_prev = load_dataset_from_disk(prev_df_path)
        except Exception as e:
            st.error("Something Went Wront when Trying to load previous data from disk.", e)
        
        return df_prev
    
    
    # @st.cache_data(ttl=3600)
    @staticmethod
    def __fetch_df_new(time_frame:str="2m")->pd.DataFrame:
                      
            df_new = pd.DataFrame()
            print(time_frame)
            with st.spinner("Loading data..."):
                df_new = load_dataset(interval=time_frame)
                # save_df(df_new, ds_path=config.DATASET_DIR)
            return df_new
            
            
    
    def __update_states(self, new_states:States):
         st.session_state.states = new_states
         
    def __extract_features(self):
         raise NotImplementedError
    
    def __plot_log_returns(self,df_featured:pd.DataFrame):
        st.header("Log returns")
        ret_cols = [ret_col for ret_col in df_featured.columns if  ret_col.startswith('log_return')]
        
        for ret_col in ret_cols:
            cols_logReturn = st.columns(2)
            figs = plot_log_returns(df_featured, col =ret_col)
            with cols_logReturn[0]:
                st.plotly_chart(figs[0],use_container_width=True)
            with cols_logReturn[1]:
                st.plotly_chart(figs[1],use_container_width=True)
    
    def __plot_volatilities(self, df_featured:pd.DataFrame):
        st.header("Volatilities ")
        volatility_cols = [vol_col for vol_col in df_featured.columns if vol_col.startswith('volatility')]
        for vol_col in volatility_cols:
            fig = plot_volatility(df_featured, col=vol_col)
            st.plotly_chart(fig, use_container_width=True)
    
    def __plot_rsi(self, df_featured:pd.DataFrame):
        st.header("RSI 14")
        
        fig = plot_rsi(df_featured)
        st.plotly_chart(fig, use_container_width=True)
    
    def __plot_macd(self, df_featured:pd.DataFrame):
        st.header("MACD")
        
        fig = plot_macd(df_featured)
        st.plotly_chart(fig, use_container_width=True)
    
    def __plot_obv(self, df_featured:pd.DataFrame):
        st.header("On-Balance Volume")
        
        fig = plot_obv(df_featured)
        st.plotly_chart(fig, use_container_width=True)
    
    def __plot_bollinger(self, df_featured:pd.DataFrame):
        st.header("Bollinger Bands with Price")
        
        fig = plot_bollinger(df_featured)
        st.plotly_chart(fig, use_container_width=True)
    
    def __plot_movingAverages(self, df_featured:pd.DataFrame):
        st.header("Multiple Simple Moving Averages with Price")
        
        fig = plot_moving_averages(df_featured)
        st.plotly_chart(fig, use_container_width=True)
    
    


# initializing state variables

# if 'states' not in st.session_state:
#     st.session_state.states= dict()
#     st.session_state.states={
#          "data_is_available": False,
#          "data_is_hidden":True,
#          "df_prev": pd.DataFrame(),
#          "df_new": pd.DataFrame(),
#          "df_new_recs": pd.DataFrame(),
#     }


# _states_ = {
#      'data_is_hidden': st.session_state.states['data_is_hidden'],
#      'data_is_available': st.session_state.states['data_is_available'],
#      'df_new' : st.session_state.states['df_new'],
#      'df_prev': st.session_state.states['df_prev'],
#      'df_new_recs': st.session_state.states['df_new_recs'],
# }


# # col1, col2 = st.sidebar.columns(2)
# if st.sidebar.button('Fetch Raw-Data'):
#     df_new,df_prev = get_data()
#     data_is_available = True 
#     update_states(_states_, [('df_new', df_new), ('df_prev', df_prev), ('data_is_available', data_is_available)])
    

# def show_raw_data():
#     st.header('Raw Data')
#     if _states_['data_is_hidden']:
#         st.write("Latest Data",_states_['df_new'])
#         st.write("Previous Data",_states_['df_prev'])
#         update_states(_states_, [('data_is_hidden', False)])
#         # st.rerun()
        
#         if _states_['df_new'].tail(1).index > _states_['df_prev'].tail(1).index:
#                 df_new_recs = _states_['df_new'][~_states_['df_new'].apply(tuple, axis=1).isin(_states_['df_prev'].apply(tuple, axis=1))]
#                 update_states(_states_, [('df_new_recs',df_new_recs)])
#                 st.warning(f"{_states_['df_new_recs'].shape[0]} New Records Are Available.")
#                 st.write("New Rows",_states_['df_new_recs'])

#     else:
#         update_states(_states_, [('data_is_hidden', True)])
#         st.rerun() # Rebuild the entire ui

# if st.sidebar.checkbox("Show Raw-Data", disabled=not _states_['data_is_available']):
#         show_raw_data()
        

# cols = st.columns(3)

# for col in cols:
#      with col:
#           st.write(f'col{col}')

data_analytics = DataAnalytics()
data_analytics.build_ui()