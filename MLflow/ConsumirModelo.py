from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

import os

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "PrediccionCalidadVinos"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

fullpath = os.path.dirname(__file__) + '/../data/'

df_test = pd.read_csv(fullpath + 'winequality-red_test.csv', sep = ',')  

x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

import mlflow

######################################################
# Consumir una ejecucion
######################################################
#model_run = '6c98b916c6954a2f89fc49777847f6d0/Regression tree'
#model = mlflow.pyfunc.load_model(model_uri=f"runs:/{model_run}")

######################################################
# Consumir una version de un modelo
######################################################
#model_name = "ModeloPuntuacionVinos"
#model_version = 2
#model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

######################################################
# Consumir un modelo en una etapa
######################################################
model_name = "ModeloPuntuacionVinos"
#model_stage = "Staging"
model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

y_tmp = y_test.copy()
y_tmp['predicted'] = model.predict(x_test)
y_tmp.columns = ['Real', 'Predicho']
        
metricas = {
    'MAE': mae(y_tmp[['Real']], y_tmp[['Predicho']]),
    'MSE': mse(y_tmp[['Real']], y_tmp[['Predicho']]),
    'RMSE': mse(y_tmp[['Real']], y_tmp[['Predicho']], squared=False),
    'MAPE': mape(y_tmp[['Real']], y_tmp[['Predicho']]),
}
print(metricas)
