from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from datetime import datetime

import shap
import matplotlib.pyplot as plt

import os

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "PrediccionCalidadVinos"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

fullpath = os.path.dirname(__file__) + '/../data/'

df_train = pd.read_csv(fullpath + 'winequality-red_train.csv', sep = ',')  
df_test = pd.read_csv(fullpath + 'winequality-red_test.csv', sep = ',')  

x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']]

x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

signature = infer_signature(x_test, y_test)

import mlflow

try:
    exp_id = mlflow.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"]).experiment_id
except Exception as e:
    exp_id = mlflow.create_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
    

dt = datetime.now().strftime("%d%b%Y-%H%M%S")

#Parametros
alpha = 0.09
#Nombre de la ejecucion
run_name = 'Regresion Lasso - Optimizacion bayesiana ' + dt
#Descripcion
description = """
"""
def new_column(x):
    x['fa_va_ratio'] = x['fixed acidity']/ x['volatile acidity']
    return x

from skopt import gp_minimize
from sklearn.metrics import mean_absolute_percentage_error as mape
import math

def objective(params):
    #unpack parameters
    alpha = params[0]

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description) as run:
        steps = [
            ('new_column', FunctionTransformer(new_column, validate=False, kw_args={})), 
            ('lasso', Lasso())
        ]
        model = Pipeline(steps)
        model.set_params(lasso__alpha=alpha)

        model.fit(x_train,y_train)

        y_tmp = y_test.copy()

        y_tmp['predicted'] = model.predict(x_test)
        y_tmp.drop('predicted', inplace=False, axis=1)
        y_tmp['index'] = range(1, len(y_tmp) + 1)

        y_tmp.columns = ['Real', 'Predicho', 'Index']
            
        metricas = {
            'MAE': mae(y_tmp[['Real']], y_tmp[['Predicho']]),
            'MSE': mse(y_tmp[['Real']], y_tmp[['Predicho']]),
            'RMSE': mse(y_tmp[['Real']], y_tmp[['Predicho']], squared=False),
            'MAPE': mape(y_tmp[['Real']], y_tmp[['Predicho']]),
        }
        print(metricas)


        ########################################################
        # Registro de parametros
        ########################################################
        mlflow.log_params({'alpha': alpha})

        ########################################################
        # Registro de métricas
        ########################################################
        mlflow.log_metrics(metricas)

        ########################################################
        # Registro del modelo
        ########################################################
        #mlflow.sklearn.log_model(model, "Lasso", signature=signature)

        ########################################################
        # Registro de artefactos: archivo fuente
        ########################################################
        #mlflow.log_artifact(__file__, artifact_path="source_code")

        ########################################################
        # Registro de tag para indicar que es una optimizacion
        ########################################################
        mlflow.set_tags({
            "TIPO DE MODELO": "OPTIMIZACION"
        })
        #mlflow.sklearn.log_model(model, "Lasso", signature=signature)
    
    return mape(y_tmp[['Real']], y_tmp[['Predicho']])


space = [
            (0.000001, 1),
        ]

r = gp_minimize(objective, space, n_calls = 15, random_state = 1)

