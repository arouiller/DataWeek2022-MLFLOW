import os
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from datetime import datetime
from skopt import gp_minimize

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']].to_numpy()

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']].to_numpy()
signature = infer_signature(x_test, y_test)

## Obtenemos el experiment id asociado a "PrediccionCalidadVinos"
experiment_name = "PrediccionCalidadVinos"
try:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except Exception as e:
    exp_id = mlflow.create_experiment(experiment_name)
    

#Nombre de la ejecucion
dt = datetime.now().strftime("%d%b%Y-%H%M%S")
run_name = 'Regresion Lasso - Optimizacion bayesiana ' + dt

print("*************************************************")
print("Ejecutando runName: " + run_name)
print("*************************************************")
#Descripcion
description = """
Optimización bayesiana para un modelo Lasso
"""
def new_column(x):
    x['fa_va_ratio'] = x['fixed acidity']/ x['volatile acidity']
    return x

iteracion = 0

def objective(params):
    global iteracion
    #unpack parameters
    alpha = params[0]
    new_run_name = run_name + '(' + str(iteracion)+ ')'
    with mlflow.start_run(experiment_id=exp_id, run_name=new_run_name, description=description ) as run:
        iteracion = iteracion + 1
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
        print("************************************************")
        print("Parametro alpha: " + "{:.6f}".format(alpha) )
        print("Metricas :")
        print(metricas)
        print("************************************************")
        print("")

        # Registro de parametros
        mlflow.log_params({'alpha': alpha})

        # Registro de métricas
        mlflow.log_metrics(metricas)

        # Registro del modelo
        #mlflow.sklearn.log_model(model, "Lasso", signature=signature)

        # Registro de artefactos
        #mlflow.log_artifact(__file__, artifact_path="source_code")

        # Registro de tag para indicar que es una optimizacion
        mlflow.set_tag("TIPO DE MODELO", "OPTIMIZACION")

        #mlflow.sklearn.log_model(model, "Lasso", signature=signature)
    
    return mape(y_test, predicciones)


space = [
            (0.000001, 1),
        ]

r = gp_minimize(objective, space, n_calls = 15, random_state = 1)

