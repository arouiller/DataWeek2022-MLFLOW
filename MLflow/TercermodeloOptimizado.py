from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


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
alpha = 0.00017204962763718543
#Nombre de la ejecucion
run_name = 'Regresion Lasso - Tercera prueba'
#Descripcion
description = """
# Descubrimiento
Añadí un nuevo campo llamado "fa_va_ratio" calculado como la división entre el campo "fixed acidity" y "volatile acidity" que mejora notablemente la predicción del modelo

El valor del hiperparámetro alpha fue obtenido a través de una optimización bayesiana en el conjunto de experimentos (Regresion Lasso - Optimizacion bayesiana 19Sep2022-085534)
"""
def new_column(x):
    x['fa_va_ratio'] = x['fixed acidity']/ x['volatile acidity']
    return x


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
    mlflow.sklearn.log_model(model, "Lasso", signature=signature)

    ########################################################
    # Registro de artefactos: archivo fuente
    ########################################################
    mlflow.log_artifact(__file__, artifact_path="source_code")

    ########################################################
    # Registro de tag para indicar que es una optimizacion
    ########################################################
    mlflow.set_tags({
        "TIPO DE MODELO": "FINAL"
    })

    #Calculo la importancia de los atributos en el modelo

    #new_x_test = model.named_steps['new_column'].transform(x_test)
    #explainer = shap.KernelExplainer(model.named_steps['lasso'].predict, new_x_test, keep_index=True)
    #shap_values = explainer.shap_values(new_x_test)
    #shap.summary_plot(shap_values, new_x_test, plot_type='violin', show=False)
    #plt.savefig('shap1.png')

    ########################################################
    # Registro de artefactos: importancia de variables
    ########################################################
    mlflow.log_artifact("shap1.png", artifact_path="img")
