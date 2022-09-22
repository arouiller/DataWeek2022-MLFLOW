import os
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import shap
import matplotlib.pyplot as plt

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']].to_numpy()

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']].to_numpy()

def new_column(x):
    x['fa_va_ratio'] = x['fixed acidity']/ x['volatile acidity']
    return x

## Obtenemos el experiment id asociado a "PrediccionCalidadVinos"
experiment_name = "PrediccionCalidadVinos"
try:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except Exception as e:
    exp_id = mlflow.create_experiment(experiment_name)

#Nombre de la ejecucion
run_name = 'Regresion Lasso - Tercera prueba'
#Descripcion
description = """
# Titulo
Optimizacion bayeasiana de los parametros
"""

with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description) as run:
    #Parametros
    alpha = 0.00017204962763718543
    steps = [
        ('new_column', FunctionTransformer(new_column, validate=False, kw_args={})), 
        ('lasso', Lasso())
    ]

    model = Pipeline(steps)
    model.set_params(lasso__alpha=alpha)

    model.fit(x_train,y_train)
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
   # Registro del modelo
    input_schema = Schema([
          ColSpec("double", "fixed acidity"),
          ColSpec("double", "volatile acidity"),
          ColSpec("double", "citric acid"),
          ColSpec("double", "residual sugar"),
          ColSpec("double", "chlorides"),
          ColSpec("double", "free sulfur dioxide"),
          ColSpec("double", "total sulfur dioxide"),
          ColSpec("double", "density"),
          ColSpec("double", "pH"),
          ColSpec("double", "sulphates"),
          ColSpec("double", "alcohol")
    ])
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    mlflow.sklearn.log_model(model, "Lasso", signature=signature)

    # Registro de artefactos
    mlflow.log_artifact(__file__, artifact_path="source_code")

    #Registro de la importancia de variables
    #explainer = shap.Explainer(model.predict, x_test)
    #shap_values = explainer(x_test)
    #shap.summary_plot(shap_values, plot_type='violin', show=False)
    #plt.savefig('shap1.png')
    #mlflow.log_artifact('shap1.png', artifact_path="documentation")

    # Registro de tag para indicar que es un modelo final
    mlflow.set_tag("TIPO DE MODELO", "FINAL")
    mlflow.set_tag("STATUS", "VALIDAR")