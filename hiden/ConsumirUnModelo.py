import os

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "PrediccionCalidadVinos"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  

x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

import mlflow.pyfunc

model_name = "PrediccionCalidadVino"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

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



