

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import config
from src.preprocessing import LoadAndPrepareData


from src.models.LogReg import LogisticModel
from src.models.CRandomForest import RandomForestClsModel
from src.models.CLGBM import LightGBMClsModel

from src.models.RRidgeReg import RidgeRegModel
from src.models.RRandomForest import RandomForestRegModel
from src.models.RLGBM import LightGBMRegModel

def show():
    st.title("🤖 Models")

    uploaded_file = st.file_uploader("Upload a CSV for modeling", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data shape:", df.shape)

        target = st.selectbox("Select target column", df.columns)

        if st.button("Train Model"):
            X = df.drop(columns=[target])
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"Model trained! Accuracy: {acc:.2f}")

st.title("🤖 Train/Load and Compare Models")


# class ModelsUI():
#     def __init__(self):
#         # print(st.session_state.states)
#         if 'states' in st.session_state:
#             self.prp:LoadAndPrepareData = st.session_state.states.prp
        
#     def buildUI(self):
#         if 'states' not in st.session_state:
#             st.warning("You Need To Open **Data Visulalization** and Fetch Data To Be Able To Train/Compare Models.")
#             return
        
        
#         prp = st.session_state.states.prp
#         if self.prp.featured_data.empty:
#                 st.warning("Please Fetch New Data from the sidebar tab **Data Visualizations**")
#                 return

#         tab_cls, tab_reg = st.tabs(["Classification Models", "Regression Models"])

#         numpy_data, targets_cls, targets_reg = prp.build_numpy_inputs_targets_data(cls_threshold=0.0002)
#         gen_cls, gen_reg = prp.build_windowed_data()
#         splited_data = prp.split_data()
#         ml_data_1h_cls = prp.splited_data["classification"]
#         ml_data_1h_reg = prp.splited_data["regression"]

#         x_train_cls = ml_data_1h_cls["x_train"]
#         x_train_cls = np.concatenate([x_train_cls, ml_data_1h_cls["x_val"]], axis=0)

#         y_train_cls = ml_data_1h_cls['y_train']
#         y_train_cls = np.concatenate([y_train_cls, ml_data_1h_cls['y_val']], axis=0)

#         y_test_cls = ml_data_1h_cls["y_test"]
#         x_test_cls = ml_data_1h_cls["x_test"]

#         x_train_reg = ml_data_1h_reg["x_train"]
#         x_train_reg = np.concatenate([x_train_reg, ml_data_1h_reg["x_val"]], axis=0)

#         y_train_reg = ml_data_1h_reg['y_train']
#         y_train_reg = np.concatenate([y_train_reg, ml_data_1h_reg['y_val']], axis=0)

#         y_test_reg = ml_data_1h_reg["y_test"]
#         x_test_reg = ml_data_1h_reg["x_test"]

#         prp.save()
        
#         with tab_cls:
            
#             st.header("Introduction")
#             st.markdown(f"Given the threshold T=.0002, and a sliding window of size 7, these Models predicts the log return of the price of the next record as, 0 if the log returned price is less than -T, 1 if it lies within the interval [-T, T] and 2 if it is greater than T.")
            
#             cls_metrics = []


#             # ============================= Logistic Regression ================================
#             if "logReg_selectBox" not in st.session_state:
#                 st.session_state.logReg_selectBox = "Use Trained Model"

#             logReg_container = st.container(key="logReg_container")

#             with logReg_container: 
#                 st.header("1. Logistic Regression")
#                 st.session_state.logReg_selectBox = st.selectbox(label= "Please Select The option For Logistic Regression", 
#                                                                  options=["Use Trained Model", "Train Model With New Data",],
#                                                                  index=["Use Trained Model", "Train Model With New Data"].index(st.session_state.logReg_selectBox),
#                                                                  key= "logReg_checkBox" )
                
#                 cls_logReg_model_:LogisticModel = None
#                 cls_logReg_metrics = None
#                 if st.session_state.logReg_selectBox == "Train Model With New Data":
#                     cls_logReg_model_ = LogisticModel(horizon=prp.interval, window_size=prp.interval)
                    
                    
#                     cls_logReg_model_.build_model({'solver': 'lbfgs', 'C': 0.1})
#                     if st.button("Train and Evaluate LR"):
#                         with st.spinner("Training the Logistic Regression for multicalss classification ... "):
#                             cls_logReg_model_.train(x_train_cls, y_train_cls)
                    
#                     if  hasattr(cls_logReg_model_, "train_date"):    
#                         with st.spinner("Evaluating the Trained Model..."):
#                             cls_logReg_metrics = cls_logReg_model_.evaluate(x_test_cls, 
#                                                                             y_test_cls, 
#                                                                             n_classes=3, 
#                                                                             task="classification")
#                         # cls_logReg_model_.save()
#                         cm = cls_logReg_metrics.pop("confusion_matrix", None)
#                         cls_metrics.append(("Logistic Regression", cls_logReg_metrics))
#                         st.html("<h2>Evaluation Metrics:</h2>")
#                         st.write(pd.DataFrame([cls_logReg_metrics]))
                        
#                         self.__plot_confusionMatrix(cm)
                
#                 elif st.session_state.logReg_selectBox == "Use Trained Model":
                    
#                     cls_logReg_model_ = LogisticModel.load(trained_models_dir=config.TRAINED_MODELS_DIR,
#                                                         horizon=self.prp.interval,
#                                                         window_size=self.prp.window_size,
#                                                         name=config.MODELS_NAMES['Logistic Regression'])
#                     with st.spinner("Evaluating the Trained Model..."):
#                         cls_logReg_metrics = cls_logReg_model_.evaluate(x_test_cls, 
#                                                                         y_test_cls, 
#                                                                         n_classes=3, 
#                                                                         task="classification")
#                     cm = cls_logReg_metrics.pop("confusion_matrix", None)
#                     cls_metrics.append(("Logistic Regression", cls_logReg_metrics))
#                     st.html("<h2>Evaluation Metrics:</h2>")
#                     st.write(pd.DataFrame([cls_logReg_metrics]))
                    
#                     self.__plot_confusionMatrix(cm)

                

#             # ============================= Random Forest ================================
#             if "regRandomForest_selectBox" not in st.session_state:
#                 st.session_state.regRandomForest_selectBox = "Use Trained Model"

#             regRandomForest_container = st.container(key="regRandomForest_container")
#             with regRandomForest_container:
#                 st.html("<hr/>")
#                 st.header("2. Random Forest Classifier")
                
#                 st.session_state.regRandomForest_selectBox= st.selectbox(label= "Please Select The option for Random Forest Classifier", 
#                                                                          options=["Use Trained Model", "Train Model With New Data", ],
#                                                                          index=["Use Trained Model", "Train Model With New Data",].index(st.session_state.regRandomForest_selectBox),
#                                                                          key="regRandomForest_checkBox")
                
#                 cls_randomForest_model_:RandomForestClsModel = None
#                 cls_randomForest_metrics = None

#                 if self.prp.featured_data.empty:
#                     st.warning("Please Fetch New Data from the sidebar tab `Data Visualizations`")
#                     raise Exception
                
#                 if st.session_state.regRandomForest_selectBox == "Train Model With New Data":
#                     cls_randomForest_model_ = RandomForestClsModel(horizon=prp.interval, window_size=prp.interval)
                    
                    
#                     cls_randomForest_model_.build_model({'n_estimators': 200, 'min_samples_split': 5, 'max_depth': 5})
                    
#                     with st.spinner("Training the Random Forest for multicalss classification ... "):
#                         cls_randomForest_model_.train(x_train_cls, y_train_cls)
                    
#                     cls_randomForest_metrics = cls_randomForest_model_.evaluate(x_test_cls, y_test_cls, n_classes=3, task="classification")
#                     # cls_randomForest_model_.save()
#                     cm = cls_randomForest_metrics.pop("confusion_matrix", None)
#                     cls_metrics.append(("Random Forest", cls_randomForest_metrics))

#                     st.html("<h2>Evaluation Metrics:</h2>")
#                     st.write(pd.DataFrame([cls_randomForest_metrics]))
                    
#                     # plotting confusion matrix
#                     self.__plot_confusionMatrix(cm)
                
#                 elif st.session_state.regRandomForest_selectBox == "Use Trained Model":
                    
#                     cls_randomForest_model_ = RandomForestClsModel.load(trained_models_dir=config.TRAINED_MODELS_DIR,
#                                                         horizon=self.prp.interval,
#                                                         window_size=self.prp.window_size,
#                                                         name=config.MODELS_NAMES['Random Forest Classifier'])
#                     with st.spinner("Loading the Trained Model..."):
#                         cls_randomForest_metrics = cls_randomForest_model_.evaluate(x_test_cls, y_test_cls, n_classes=3, task="classification")
#                     cm = cls_randomForest_metrics.pop("confusion_matrix", None)
#                     cls_metrics.append(("Random Forest", cls_randomForest_metrics))

#                     st.html("<h2>Evaluation Metrics:</h2>")
#                     st.write(pd.DataFrame([cls_randomForest_metrics]))
                    
#                     # plotting confusion matrix
#                     self.__plot_confusionMatrix(cm)
        
#             # ============================= LGBM ================================
#             if "regLGBM_selectBox" not in st.session_state:
#                 st.session_state.regLGBM_selectBox = "Use Trained Model"

#             regLGBM_container = st.container(key="regLGBM_container")
#             with regLGBM_container:

#                 st.html("<hr/>")
#                 st.header("3. LightGBM (Light Gradient Boosting Machine) ")
#                 st.session_state.regLGBM_selectBox  = st.selectbox(label= "Please Select one of The options for LightGBM Classifier", 
#                                                                    options=["Train Model With New Data", "Use Trained Model"],
#                                                                    index=["Train Model With New Data", "Use Trained Model"].index(st.session_state.regLGBM_selectBox),
#                                                                    key="regLGBM_checkBox")
                
#                 cls_LGBM_model_:LightGBMClsModel = None
#                 cls_LGBM_metrics = None

#                 if self.prp.featured_data.empty:
#                     st.warning("Please Fetch New Data from the sidebar tab `Data Visualizations`")
#                     raise Exception
                
        
#                 if st.session_state.regLGBM_selectBox == "Train Model With New Data":
#                     cls_LGBM_model_ = LightGBMClsModel(horizon=prp.interval, window_size=prp.interval)
                    
                    
#                     cls_LGBM_model_.build_model({
#                                                 "num_leaves": 15,
#                                                 "max_depth": 5,
#                                                 "learning_rate": 0.01,
#                                                 "n_estimators": 300,
#                                                 "subsample":0.6,
#                                                 "colsample_bytree": 0.6,
#                                                 "reg_alpha": 0.5,
#                                                 "reg_lambda":  0.5
#                                                 })
#                     if st.button("Train and Evaluate LGBM"):
#                         with st.spinner("Training the LightGBM for multicalss classification ... "):
#                             cls_LGBM_model_.train(x_train_cls, y_train_cls)
                        
#                         cls_LGBM_metrics = cls_LGBM_model_.evaluate(x_test_cls, y_test_cls, n_classes=3, task="classification")
                        
                        
#                         cm = cls_LGBM_metrics.pop("confusion_matrix", None)
#                         cls_metrics.append(("LightGBM", cls_LGBM_metrics))
#                         metrics_df = pd.DataFrame([cls_LGBM_metrics])
#                         st.html("<h2>Evaluation Metrics:</h2>")
#                         st.write(metrics_df)

#                         # plotting confusion matrix
#                         self.__plot_confusionMatrix(cm)
#                         # _, centerCol, _ = st.columns(3, vertical_alignment="center")
#                         # with centerCol:
#                             # if st.button("Save The Trained LGBM To Disk", type="primary" ):
#                                 # cls_LGBM_model_.save()
#                         # cls_LGBM_model_.save()
                
#                 elif st.session_state.regLGBM_selectBox == "Use Trained Model":
                    
#                     cls_LGBM_model_ = LightGBMClsModel.load(trained_models_dir=config.TRAINED_MODELS_DIR,
#                                                         horizon=self.prp.interval,
#                                                         window_size=self.prp.window_size,
#                                                         name=config.MODELS_NAMES['LGBM Classifier'])

#                     with st.spinner("Loading the Trained Model..."):
#                         cls_LGBM_metrics = cls_LGBM_model_.evaluate(x_test_cls, y_test_cls, n_classes=3, task="classification")
#                     st.html("<h2>Evaluation Metrics:</h2>")
#                     cm = cls_LGBM_metrics.pop("confusion_matrix", None)
#                     metrics_df = pd.DataFrame([cls_LGBM_metrics])
#                     cls_metrics.append(("LightGBM", cls_LGBM_metrics))
#                     st.write(metrics_df)

#                     self.__plot_confusionMatrix(cm)

#             # ============================= Comparing Classifiers ================================
#             st.html("<hr/>")
#             st.header("Comparing Models")
#             # Extract metric names
#             if len(cls_metrics)>1 :
                
#                 metrics = [metric for metric in cls_metrics[0][1]]
#                 metrics.remove("train_date")
#                 # Create traces for each model
#                 fig = go.Figure()
#                 for model_name, scores in cls_metrics:
#                     fig.add_trace(go.Bar(
#                         x= [str.upper(metric) for metric in cls_metrics[0][1]],
#                         y=[scores[m] for m in metrics],
#                         name=model_name
#                     ))

#                 # Layout
#                 fig.update_layout(
#                     title="Model Comparison Across Metrics",
#                     xaxis_title="Metric",
#                     yaxis_title="Score",
#                     barmode='group',
#                     height=500
#                 )

#                 # Display in Streamlit
#                 st.plotly_chart(fig)
#             else:
#                 st.warning("You Need to Train/Import at least 2 Models to Compare", icon="⚠️")

#         with tab_reg:
#             st.header("Introduction")
#             st.markdown("\
#             Information")


#             reg_metrics = []
#             # ============================= Ridge Regression ================================
            
#             st.header("1. Ridge Regression")
#             RR_trainOrLoad_selectBox = st.selectbox(label= "Please Select The option for Ridge Regression", options=["Train Model With New Data", "Use Trained Model"])
            
#             reg_ridgeReg_model_:RidgeRegModel = None
#             reg_ridgeReg_metrics = None

#             if self.prp.featured_data.empty:
#                 st.warning("Please Fetch New Data from the sidebar tab `Data Visualizations`")
#                 raise Exception
            
    
#             if RR_trainOrLoad_selectBox == "Train Model With New Data":
#                 reg_ridgeReg_model_ = RidgeRegModel(horizon=prp.interval, window_size=prp.interval)
                
                
#                 reg_ridgeReg_model_.build_model({'solver': 'lsqr', 'alpha': 0.1})
#                 if st.button("Train and Evaluate Ridge Regression"):
#                     with st.spinner("Training Ridge Regression for Predicting Feture Log Return ... "):
#                         reg_ridgeReg_model_.train(x_train_reg, y_train_reg)
                    
#                     reg_ridgeReg_metrics = reg_ridgeReg_model_.evaluate(x_test_reg, y_test_reg, task="regression")
#                     reg_ridgeReg_model_.save()
#                     reg_metrics.append(("Ridge Regression", reg_ridgeReg_metrics))

#                     st.html("<h2>Evaluation Metrics:</h2>")
#                     st.write(pd.DataFrame([reg_ridgeReg_metrics]))
                    
#                     # plotting models prediction vs true values
#                     y_pred = reg_ridgeReg_model_.predict(x_test_reg)
#                     self.__plot_predTrue(y_test_reg, y_pred)
            
#             elif RR_trainOrLoad_selectBox == "Use Trained Model":
                
#                 reg_ridgeReg_model_ = RidgeRegModel.load(trained_models_dir=config.TRAINED_MODELS_DIR,
#                                                     horizon=self.prp.interval,
#                                                     window_size=self.prp.window_size,
#                                                     name=config.MODELS_NAMES['Ridge Regression'])
#                 with st.spinner("Loading the Trained Model..."):
#                     reg_ridgeReg_metrics = reg_ridgeReg_model_.evaluate(x_test_reg, y_test_reg, task="regression")
#                 cls_metrics.append(("Random Forest", reg_ridgeReg_metrics))

#                 st.html("<h2>Evaluation Metrics:</h2>")
#                 st.write(pd.DataFrame([reg_ridgeReg_metrics]))
                
#                 # plotting models prediction vs true values
#                 y_pred = reg_ridgeReg_model_.predict(x_test_reg)
#                 self.__plot_predTrue(y_test_reg, y_pred)


from typing import Dict
from streamlit_app.src.models.Model import Model


MODELS = {
        config.MODELS_NAMES["Logistic Regression"]: LogisticModel,
        config.MODELS_NAMES["Random Forest Classifier"]: RandomForestClsModel,
        config.MODELS_NAMES["LGBM Classifier"]: LightGBMClsModel,
        
        config.MODELS_NAMES["Ridge Regression"]: RidgeRegModel,
        config.MODELS_NAMES["Random Forest Regressor"]:RandomForestRegModel,
        config.MODELS_NAMES["LGBM Regressor"]:LightGBMRegModel
    }
rev_MODELS_NAMES={
    val: key for key, val in config.MODELS_NAMES.items()
}
class ModelsUI:
    def __init__(self):
        if 'states' in st.session_state:
            self.prp: LoadAndPrepareData = st.session_state.states.prp
        else:
            st.warning("Please fetch data from **Data Visualization** on sidebar")
            return
        numpy_data, targets_cls, targets_reg = self.prp.build_numpy_inputs_targets_data(cls_threshold=0.0002)
        gen_cls, gen_reg = self.prp.build_windowed_data()
        splited_data = self.prp.split_data()

        if 'models' not in st.session_state:
            st.session_state.models = dict()
       

    def buildUI(self):
        if 'states' not in st.session_state:
            st.warning("You Need To Open **Data Visualization** and Fetch Data First.")
            return
        
        self.prp = st.session_state.states.prp
        tab_cls, tab_reg = st.tabs(["Classification Models", "Regression Models"])

        x_train_cls, y_train_cls, x_test_cls, y_test_cls,\
        x_train_reg, y_train_reg, x_test_reg, y_test_reg = self.__get_modelFeed()
        

        with tab_cls:
            
            clsMetrics = []
            logReg_clsMetrics = self.render_clsModel(config.MODELS_NAMES["Logistic Regression"],
                                                    {'solver': 'lbfgs', 'C': 0.1},
                                                    x_train_cls, 
                                                    x_test_cls, 
                                                    y_train_cls, 
                                                    y_test_cls,)
            if logReg_clsMetrics:
                clsMetrics.append(("Logistic Regression",logReg_clsMetrics))
            st.html("<hr style='height:30px;border-width:0;color:green;background-color:green'/>")
            randomForest_clsMetrics = self.render_clsModel(config.MODELS_NAMES["Random Forest Classifier"],
                                                    {'n_estimators': 200, 'min_samples_split': 2, 'max_depth': 5},
                                                    x_train_cls, 
                                                    x_test_cls, 
                                                    y_train_cls, 
                                                    y_test_cls,)
            if randomForest_clsMetrics:
                clsMetrics.append(("Random Forest", randomForest_clsMetrics))
            st.html("<hr style='height:30px;border-width:0;color:green;background-color:green'/>")
            LGBM_clsMetrics = self.render_clsModel(config.MODELS_NAMES["LGBM Classifier"],
                                                    {
                                                    "num_leaves": 15,
                                                    "max_depth": 5,
                                                    "learning_rate": 0.01,
                                                    "n_estimators": 300,
                                                    "subsample":0.6,
                                                    "colsample_bytree": 0.6,
                                                    "reg_alpha": 0.5,
                                                    "reg_lambda":  0.5
                                                    },
                                                    x_train_cls, 
                                                    x_test_cls, 
                                                    y_train_cls, 
                                                    y_test_cls,)
            if LGBM_clsMetrics:
                clsMetrics.append(("LGBM", LGBM_clsMetrics))
            st.html("<hr style='height:30px;border-width:0;color:green;background-color:green'/>")
            
            self.compare_clsModels(clsMetrics)
            # self.render_lgbm_cls()
            # self.compare_classifiers()

        with tab_reg:
            reg_metrics = []
            ridge_regMetrics = self.render_regModel(
                                                    config.MODELS_NAMES["Ridge Regression"],
                                                    {'solver': 'lsqr', 'alpha': 0.1},
                                                    x_train_reg,
                                                    x_test_reg,
                                                    y_train_reg,
                                                    y_test_reg)
            if ridge_regMetrics:
                reg_metrics.append(("Ridge Regression", ridge_regMetrics))
            
            randomForest_regMetrics = self.render_regModel(
                                                    config.MODELS_NAMES["Random Forest Regressor"],
                                                    {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 5},
                                                    x_train_reg,
                                                    x_test_reg,
                                                    y_train_reg,
                                                    y_test_reg)
            if randomForest_regMetrics:
                reg_metrics.append(("Random Forest", randomForest_regMetrics))
            
            LGBM_regMetrics = self.render_regModel(
                                                    config.MODELS_NAMES["LGBM Regressor"],
                                                    {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 5},
                                                    x_train_reg,
                                                    x_test_reg,
                                                    y_train_reg,
                                                    y_test_reg)
            if LGBM_regMetrics:
                reg_metrics.append(("LGBM", LGBM_regMetrics))
            self.compare_regModels(reg_metrics)
            
                                        
    def render_clsModel(self, 
                        model_name:str,
                        hyper_params, 
                        x_train, 
                        x_test, 
                        y_train, 
                        y_test):
        metrics = dict()
        st.subheader(rev_MODELS_NAMES[model_name])
        option = st.selectbox("Choose option for "+rev_MODELS_NAMES[model_name],
                                    ["Use Trained Model", "Train Model With New Data"],
                                    key=rev_MODELS_NAMES[model_name]+"_option")
        if option == "Train Model With New Data":
            bt_ = st.button("Retrain Model", key=model_name+"_reTrainBtn")
            model = MODELS[model_name](horizon=self.prp.interval, window_size=self.prp.interval)
            if bt_:
                model.build_model(hyper_params)
                with st.spinner(f"Training {rev_MODELS_NAMES[model_name]} ... "):
                    model.train(x_train, y_train)
                st.session_state.models[model_name] = model
                
                model = st.session_state.models[model_name]
                metrics = model.evaluate(x_test,
                                        y_test,
                                        n_classes=3,
                                        task="classification")
                cm = metrics.pop("confusion_matrix")
                # _ = metrics.pop("train_date")
                st.write(pd.DataFrame([metrics]))
                self.__plot_confusionMatrix(cm)
            
            if model_name in st.session_state.models and not bt_:
                model = st.session_state.models[model_name]
                metrics = model.evaluate(x_test,
                                        y_test,
                                        n_classes=3,
                                        task="classification")
                cm = metrics.pop("confusion_matrix")
                # _ = metrics.pop("train_date")
                st.write(pd.DataFrame([metrics]))
                self.__plot_confusionMatrix(cm)
                

        else:
            model = MODELS[model_name].load(trained_models_dir=config.TRAINED_MODELS_DIR,
                                        horizon=self.prp.interval,
                                        window_size=self.prp.window_size,
                                        name=model_name)
            metrics = model.evaluate(x_test,
                                    y_test,
                                    n_classes=3,
                                    task="classification")
            cm = metrics.pop("confusion_matrix")
            # _ = metrics.pop("train_date")
            st.write(pd.DataFrame([metrics]))
            self.__plot_confusionMatrix(cm)
        
        return metrics

    def render_regModel(self, 
                        model_name:str,
                        hyper_params, 
                        x_train, 
                        x_test, 
                        y_train, 
                        y_test):
        metrics = dict()
        st.subheader(rev_MODELS_NAMES[model_name])
        option = st.selectbox("Choose option for "+rev_MODELS_NAMES[model_name],
                              ["Use Trained Model", "Train Model With New Data"],
                              key=rev_MODELS_NAMES[model_name]+"_option")
        
        if option == "Train Model With New Data":
            bt_ = st.button("Retrain Model", key=model_name+"_reTrainBtn")
            model = MODELS[model_name](horizon=self.prp.interval, window_size=self.prp.interval)
            if bt_:
                model.build_model(hyper_params)
                with st.spinner(f"Training {rev_MODELS_NAMES[model_name]} ... "):
                    model.train(x_train, y_train)
                st.session_state.models[model_name] = model

                model = st.session_state.models[model_name]
                metrics = model.evaluate(x_test, y_test, task="regression")
                # _ = metrics.pop("train_date", None)
                st.write(pd.DataFrame([metrics]))
                y_pred = model.predict(x_test)
                self.__plot_predTrue(y_test, y_pred)

            if model_name in st.session_state.models and not bt_:
                model = st.session_state.models[model_name]
                metrics = model.evaluate(x_test, y_test, task="regression")
                # _ = metrics.pop("train_date", None)
                st.write(pd.DataFrame([metrics]))
                y_pred = model.predict(x_test)
                self.__plot_predTrue(y_test, y_pred)

        else:
            model = MODELS[model_name].load(trained_models_dir=config.TRAINED_MODELS_DIR,
                                            horizon=self.prp.interval,
                                            window_size=self.prp.window_size,
                                            name=model_name)
            metrics = model.evaluate(x_test, y_test, task="regression")
            # _ = metrics.pop("train_date", None)
            st.write(pd.DataFrame([metrics]))
            y_pred = model.predict(x_test)
            self.__plot_predTrue(y_test, y_pred)

        return metrics


    def compare_clsModels(self, cls_metrics):
        st.subheader("📊 Compare Classification Models")
        if len(cls_metrics) < 2:
            st.info("Train at least 2 models to compare.")
            return
        
        # Extract metric names
                 
        metrics = [metric for metric in cls_metrics[0][1]]
        metrics.remove("train_date")
        # Create traces for each model
        fig = go.Figure()
        for model_name, scores in cls_metrics:
            fig.add_trace(go.Bar(
                x= [str.upper(metric) for metric in cls_metrics[0][1]],
                y=[scores[m] for m in metrics],
                name=model_name
            ))

        # Layout
        fig.update_layout(
            title="Model Comparison Across Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode='group',
            height=500
        )

        # Display in Streamlit
        st.plotly_chart(fig)
        
    
    def compare_regModels(self, reg_metrics):
        st.subheader("📊 Compare Regression Models")
        if len(reg_metrics) < 2:
            st.info("Train at least 2 models to compare.")
            return
        
        # Extract metric names
                 
        metrics = [metric for metric in reg_metrics[0][1]]
        metrics.remove("train_date")
        # Create traces for each model
        fig = go.Figure()
        for model_name, scores in reg_metrics:
            fig.add_trace(go.Bar(
                x= [str.upper(metric) for metric in reg_metrics[0][1]],
                y=[scores[m] for m in metrics],
                name=model_name
            ))

        # Layout
        fig.update_layout(
            title="Model Comparison Across Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode='group',
            height=500
        )

        # Display in Streamlit
        st.plotly_chart(fig)

    def __get_modelFeed(self):
        

        ml_data_1h_cls = self.prp.splited_data["classification"]
        ml_data_1h_reg = self.prp.splited_data["regression"]

        x_train_cls = ml_data_1h_cls["x_train"]
        x_train_cls = np.concatenate([x_train_cls, ml_data_1h_cls["x_val"]], axis=0)

        y_train_cls = ml_data_1h_cls['y_train']
        y_train_cls = np.concatenate([y_train_cls, ml_data_1h_cls['y_val']], axis=0)

        y_test_cls = ml_data_1h_cls["y_test"]
        x_test_cls = ml_data_1h_cls["x_test"]

        x_train_reg = ml_data_1h_reg["x_train"]
        x_train_reg = np.concatenate([x_train_reg, ml_data_1h_reg["x_val"]], axis=0)

        y_train_reg = ml_data_1h_reg['y_train']
        y_train_reg = np.concatenate([y_train_reg, ml_data_1h_reg['y_val']], axis=0)

        y_test_reg = ml_data_1h_reg["y_test"]
        x_test_reg = ml_data_1h_reg["x_test"]

        

        return x_train_cls, y_train_cls, x_test_cls, y_test_cls, x_train_reg, y_train_reg, x_test_reg, y_test_reg

    def __plot_confusionMatrix(self, cm):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Random Forest)")

        st.pyplot(fig)

    def __plot_predTrue(self, y_true, y_pred):
        prp = st.session_state.states.prp
        test_startIndex = np.argwhere(prp.target_reg == y_true[0])
        dates = prp.input_data_dateTime[test_startIndex[0][0]:]
        # --- Plotly figure ---
        fig = go.Figure()

        # Add true values
        fig.add_trace(go.Scatter(
            x=dates, y=y_true,
            mode="lines+markers",
            name="y_true",
            line=dict(color="blue")
        ))

        # Add predicted values
        fig.add_trace(go.Scatter(
            x=dates, y=y_pred,
            mode="lines+markers",
            name="y_pred",
            line=dict(color="red", dash="dash")
        ))

        # Layout
        fig.update_layout(
            title="Regression Model: y_true vs y_pred",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
        )

        # --- Streamlit display ---
        st.plotly_chart(fig, use_container_width=True)
        

modelsUI =ModelsUI()

modelsUI.buildUI()