import os
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "PrediccionCalidadVinos"

fullpath = os.path.dirname(os.path.abspath(__file__)) + '/../data/'
df_test = pd.read_csv(fullpath + 'winequality-red_test.csv', sep = ',') 
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']].to_numpy()

#Consumir una version de un modelo
model_name = "ModeloPuntuacionVinos"

for model_version in range(1, 5):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    predicciones = model.predict(x_test)
    
    metricas = {
        'MAE': mae(y_test, predicciones),
        'MSE': mse(y_test, predicciones),
        'RMSE': mse(y_test, predicciones, squared=False),
        'MAPE': mape(y_test, predicciones)
    }

    print('********************************************************************************************************************************************')
    print('Version: ' + str(model_version))
    print(metricas)
    print('********************************************************************************************************************************************')
    print('')