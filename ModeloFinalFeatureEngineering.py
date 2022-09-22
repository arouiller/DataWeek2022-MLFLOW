from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape
import joblib

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
df_train['new_column'] = df_train['fixed acidity']/df_train['volatile acidity']
x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']]

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  
df_test['new_column'] = df_test['fixed acidity']/df_test['volatile acidity']
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

alpha = 0.1
model = Lasso(alpha=alpha).fit(x_train,y_train)

predicciones = model.predict(x_test)
    
metricas = {
    'MAE': mae(y_test[['quality']], predicciones),
    'MSE': mse(y_test[['quality']], predicciones),
    'RMSE': mse(y_test[['quality']], predicciones, squared=False),
    'MAPE': mape(y_test[['quality']], predicciones)
}

print(metricas)

#joblib.dump(model,  './modelos/CalificacionVinos.pkl')


# Realizando una división entre fixed acidity y volatile acidity se obtienen mejores resultados con el uso de los mismos hiperparametros
# 'MAE' : 0.532
# 'MSE' : 0.450
# 'RMSE': 0.671
# 'MAPE': 0.098




