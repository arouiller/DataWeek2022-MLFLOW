import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape

import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']].to_numpy()

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']].to_numpy()

## Obtenemos el experiment id asociado a "PrediccionCalidadVinos"
experiment_name = "PrediccionCalidadVinos"
try:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except Exception as e:
    exp_id = mlflow.create_experiment(experiment_name)

#Nombre de la ejecucion
run_name = 'Regresion Lasso - Primera prueba'
#Descripcion
description = """
Esta es la primera ejecución que realizo
"""

with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description) as run:
    alpha = 0.1
    model = Lasso(alpha=alpha).fit(x_train,y_train)
    predicciones = model.predict(x_test)
        
    metricas = {
        'MAE': mae(y_test, predicciones),
        'MSE': mse(y_test, predicciones),
        'RMSE': mse(y_test, predicciones, squared=False),
        'MAPE': mape(y_test, predicciones)
    }

    print(metricas)
    
    # Registro de parametros
    mlflow.log_params({'alpha': alpha})

    # Registro de métricas
    mlflow.log_metrics(metricas)

    # Registro del modelo
    mlflow.sklearn.log_model(model, "Lasso")

    # Registro de artefactos
    mlflow.log_artifact(__file__, artifact_path="source_code")

    # Registro de tag para indicar que es un modelo final
    mlflow.set_tag("TIPO DE MODELO", "FINAL")
    mlflow.set_tag("STATUS", "VALIDAR")