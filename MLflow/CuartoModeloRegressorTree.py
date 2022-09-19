from sklearn.tree import DecisionTreeRegressor
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from mlflow.models.signature import infer_signature

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
    
#Parametros
#
#Nombre de la ejecucion
run_name = 'Regression Tree - Quinta prueba'
#Descripcion
description = """
# Descripcion
Se utilizó una regresión con un árbol de decisión con parámetros por defecto y sin nuevas variables
"""

with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description) as run:

    model = DecisionTreeRegressor(random_state=0)
    model.fit(x_train, y_train)

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
    #mlflow.log_params({'alpha': alpha})

    ########################################################
    # Registro de métricas
    ########################################################
    mlflow.log_metrics(metricas)

    ########################################################
    # Registro del modelo
    ########################################################
    mlflow.sklearn.log_model(model, "Regression tree", signature=signature)

    ########################################################
    # Registro de artefactos: archivo fuente
    ########################################################
    mlflow.log_artifact(__file__, artifact_path="source_code")

    #Calculo la importancia de los atributos en el modelo
    explainer = shap.Explainer(model.predict, x_test)
    shap_values = explainer(x_test)
    shap.summary_plot(shap_values, plot_type='violin', show=False)
    plt.savefig('shap1.png')

    ########################################################
    # Registro de artefactos: importancia de variables
    ########################################################
    mlflow.log_artifact("shap1.png", artifact_path="img")

    ########################################################
    # Registro de tag para indicar que es una optimizacion
    ########################################################
    mlflow.set_tags({
        "TIPO DE MODELO": "FINAL"
    })