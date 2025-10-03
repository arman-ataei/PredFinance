# config.py
import os

# Directories
PROJECT_DIR = os.getcwd()
APP_DIR =  os.path.join(PROJECT_DIR, 'streamlit_app')
SRC_DIR = os.path.join(APP_DIR, 'src')

DATASET_DIR = os.path.join(SRC_DIR, 'dataset')
LOAD_AND_PREPARE_DIR = os.path.join(DATASET_DIR, 'LoadAndPrepareData')
# DBM_DIR = os.path.join(SRC_DIR, 'dbm') # database management DIRECTORY
# DATABASE_PATH = os.path.join(SRC_DIR, 'btc.db')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
TRAINED_MODELS_DIR = os.path.join(SRC_DIR, 'trained_models') 

# Database


# Models 
MODELS = ['C_LogReg', 'C_RandomForest', 'C_LGBM', 'R_RidgeReg', 'R_RandomForest', 'R_LGBM']
MODELS_NAMES={
    "Logistic Regression": 'C_LogReg',
    "Random Forest Classifier": 'C_RandomForest',
    "LGBM Classifier": 'C_LGBM',
    
    "Ridge Regression": "R_RidgeReg",
    "Random Forest Regressor": 'R_RandomForest',
    "LGBM Regressor": 'R_LGBM',
}
# 