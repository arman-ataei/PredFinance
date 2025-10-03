

from streamlit_app.utils import get_now
from streamlit_app import config

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from .Model import Model


class LightGBMClsModel(Model):
    def __init__(self, horizon = '5m', window_size = 5, save_dir = config.TRAINED_MODELS_DIR):
        super().__init__("C_LGBM", horizon, window_size, save_dir)
    
    def build_model(self, hyper_params):
        try :
            self.model = lgb.LGBMClassifier(**hyper_params)
        except Exception as e:
             print(e)
    
    def train(self, X_train, y_train):
        self.train_date = get_now()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def tune_hyperparameters(self, 
                             X_train, 
                             y_train, 
                             param_distributions, 
                             n_iter,
                             early_stopping_rounds:int=20):
        self.train_date = get_now()
        random_search = RandomizedSearchCV(
                                            lgb.LGBMClassifier(),
                                            param_distributions=param_distributions,
                                            n_iter=n_iter,          # تعداد نمونه‌گیری تصادفی
                                            cv=3,
                                            scoring="accuracy",
                                            # early_stopping_rounds=early_stopping_rounds,
                                            n_jobs=-1,
                                            random_state=42
                                        )
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        return self.best_params

# def update(self, X_new, y_new):
#     """
#     Update model with new data.

#     """
#     if self.model is None:
#         print("There is no model to update, please build one first.")
#     else:
#         self.model.set_params(warm_start=True)
#         self.model.fit(X_new, y_new, init_model=self.model)

def update(self, X_new, y_new):
        # LightGBM → continue training with keep_training_booster
        self.model.fit(X_new, y_new, init_model=self.model.booster_, keep_training_booster=True)

