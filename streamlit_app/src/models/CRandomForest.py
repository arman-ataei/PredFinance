
from streamlit_app.utils import get_now
from streamlit_app import config


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from .Model import Model



class RandomForestClsModel(Model):
    def __init__(self, horizon = '5m', window_size = 5, save_dir = config.TRAINED_MODELS_DIR):
        super().__init__("C_RandomForest", horizon, window_size, save_dir)

    def build_model(self, hyper_params):
        self.model = RandomForestClassifier(warm_start=True, **hyper_params)

    def train(self, X_train, y_train):
        self.train_date = get_now()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    
    def tune_hyperparameters(self, X_train, y_train, param_distributions, n_iter=20):
        search = RandomizedSearchCV(
            RandomForestClassifier(warm_start=True),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        return self.best_params
    

    def grid_tune_hyperparameters(self, X_train, y_train, param_grid):
        self.train_date = get_now()
        grid = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        self.model = grid.best_estimator_
        return self.best_params

    def update(self, X_new, y_new):
        # warm_start → افزایش تعداد درخت‌ها
        self.model.n_estimators += 10
        self.model.fit(X_new, y_new)

