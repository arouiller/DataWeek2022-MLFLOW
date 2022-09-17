from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

df_train = pd.read_csv('./data/winequality-red_train.csv', sep = ',')  
df_test = pd.read_csv('./data/winequality-red_test.csv', sep = ',')  

df_train['new_column'] = df_train['fixed acidity']/df_train['volatile acidity']
df_test['new_column'] = df_test['fixed acidity']/df_test['volatile acidity']

x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']]

x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

alpha = 0.1
model = Lasso(alpha=alpha).fit(x_train,y_train)

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

# Realizando una división entre fixed acidity y volatile acidity se obtienen mejores resultados con el uso de los mismos hiperparametros
# 'MAE' : 0.532
# 'MSE' : 0.450
# 'RMSE': 0.671
# 'MAPE': 0.098




