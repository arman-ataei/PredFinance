
from streamlit_app.utils import get_now
from streamlit_app import config

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from .Model import Model


class LightGBMRegModel(Model):
    def __init__(self, horizon = '5m', window_size = 5, save_dir = config.TRAINED_MODELS_DIR):
        super().__init__("R_LGBM", horizon, window_size, save_dir)

    def build_model(self, hyper_params):
        self.model = lgb.LGBMRegressor(**hyper_params)

    def train(self, X_train, y_train):
        self.train_date = get_now()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def tune_hyperparameters(self, X_train, y_train, param_distributions, n_iter=20):
        self.train_date = get_now()
        search = RandomizedSearchCV(
            lgb.LGBMRegressor(),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        return self.best_params

    def update(self, X_new, y_new):
        self.model.fit(X_new, y_new, init_model=self.model.booster_, keep_training_booster=True)
