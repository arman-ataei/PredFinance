
# Model.py
import os
import json
import pickle
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Literal

from  streamlit_app import config
from streamlit_app.utils import get_now

import numpy as np
from pandas import DataFrame
from sklearn.metrics import (accuracy_score,
                             f1_score, 
                             roc_auc_score, 
                             mean_absolute_error, 
                             mean_squared_error)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

# -------------------------------
# Base Class
# -------------------------------
class Model(ABC):
    def __init__(self, 
                 name: str,
                 horizon: str='5m',
                 window_size:int=5,
                 save_dir: str = config.TRAINED_MODELS_DIR):
        self.name:str = name
        self.horizon:str = horizon
        self.save_dir:str  = os.path.join(save_dir, horizon)
        
        os.makedirs(self.save_dir, exist_ok=True)


        self.window_size:int = window_size
        self.model = None
        self.best_params = None
        self.metrics:Dict = dict

    @abstractmethod
    def build_model(self, hyper_params: Dict[str, Any]):
        """Build model with given hyperparameters"""
        pass

    @abstractmethod
    def train(self, 
              X_train, 
              y_train,):
        """Train the model"""
        self.train_date = get_now()
        pass

    @abstractmethod
    def predict(self, X):
        """Predict outputs"""
        pass

    @abstractmethod
    def tune_hyperparameters(self, 
                            X_train, 
                            y_train,
                            param_grid: Dict[str, list]):
        """Search over hyperparameter grid and set best_params"""
        pass
    
    
    def update(self, X_new, y_new):
        """
        Update model with new data.
        - If model supports partial_fit → incremental update
        - If model has warm_start → extend training
        - Otherwise → retrain from scratch
        """
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X_new, y_new)
        elif hasattr(self.model, "warm_start") and self.model.warm_start:
            # مثال برای RandomForest: افزایش تعداد درخت‌ها
            if hasattr(self.model, "n_estimators"):
                self.model.n_estimators += 10
            self.model.fit(X_new, y_new)
        else:
            # fallback: retrain
            self.model.fit(X_new, y_new)
    
    def evaluate(self, 
                 X, 
                 y, 
                 n_classes:int=3,
                 task:str = Literal["classification", "regression"]):
        """Evaluate model and compute metrics"""
        assert n_classes>2 

        y_pred = self.predict(X)
        if task == "classification":
            acc = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            if n_classes == 2:
                f1 = f1_score(y, y_pred)
                auc = roc_auc_score(y, y_pred)
            else:
                f1 = f1_score(y, y_pred, average="weighted")
                try:
                    y_score = self.model.predict_proba(X)
                    y_bin = label_binarize(y, classes=list(range(n_classes)))
                    auc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
                except:
                    auc = None
            self.metrics = {"accuracy": acc, "f1": f1, "auc": auc, "confusion_matrix": cm}
        else:
            range_ = np.max(y) - np.min(y)
            rmse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            self.metrics = {"rmse": rmse, 'mae': mae, 'range': range_}
        
        self.metrics['train_date'] = self.train_date if hasattr(self, "train_date") else ""
        return self.metrics

    def save(self):
        """Save model and metrics"""
        # Model Directory
        model_dir = os.path.join(self.save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        # Archive Directory of the Model
        archive_dir = os.path.join(model_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        # The Model
        model_path = os.path.join(model_dir, f"{self.name}_{self.window_size}.pkl")
        
        # Archive previous version
        if os.path.exists(model_path):
            created_time = os.path.getctime(model_path)
            readable_time = datetime.datetime.fromtimestamp(created_time)
            os.rename(model_path, os.path.join(archive_dir, f"{self.name}_{self.window_size}_{readable_time}.pkl"))
        
        # Save with pickle
        try:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(e)
        # # Saving the model
        # joblib.dump(self, model_path)
        
        # # metrics
        # metrics_path = os.path.join(self.save_dir, f"{self.name}__{self.horizon}h_{self.train_date or get_now()}_metrics.json")
        # prev_metrics_path = filter(os.listdir(model_dir), lambda x: x.endswith("json"))
        # print(prev_metrics_path)
        
        # # moving prev metrics_path to the archive directory
        # if os.path.exists(prev_metrics_path):
        #     os.rename(prev_metrics_path, os.path.join(archive_dir, os.path.basename(prev_metrics_path)))    
        
        # # Saving the metrics
        # with open(metrics_path, "w") as f:
        #     json.dump(self.metrics, f, indent=2)
    @staticmethod
    def load(trained_models_dir:str=config.TRAINED_MODELS_DIR,
             horizon:str='5m',
             name:str=Literal[*config.MODELS],
             window_size:int=5 ):
        """Load model"""
        model_path = os.path.join(trained_models_dir, horizon,name, f"{name}_{window_size}.pkl" )
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        return obj

        # joblib.load(os.path.join(self.save_dir, f"{self.name}_{self.horizon}h.joblib"))
