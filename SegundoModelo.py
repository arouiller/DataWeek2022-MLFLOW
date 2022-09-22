from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, mean_absolute_percentage_error as mape

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']]

df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  
x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

alpha = 0.6
model = Lasso(alpha=alpha).fit(x_train,y_train)

predicciones = model.predict(x_test)
    
metricas = {
    'MAE': mae(y_test[['quality']], predicciones),
    'MSE': mse(y_test[['quality']], predicciones),
    'RMSE': mse(y_test[['quality']], predicciones, squared=False),
    'MAPE': mape(y_test[['quality']], predicciones)
}

print(metricas)

# 'MAE' : 0.657
# 'MSE' : 0.632
# 'RMSE': 0.795
# 'MAPE': 0.121


