from sklearn.linear_model import Lasso
import pandas as pd
import joblib
import os

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

fullpath = os.path.dirname(__file__) + '/../data/'
storing_path = os.path.dirname(__file__) + '/modelos/'

df_train = pd.read_csv(fullpath + 'winequality-red_train.csv', sep = ',')  
df_test = pd.read_csv(fullpath + 'winequality-red_test.csv', sep = ',')  

x_train = df_train.drop('quality', inplace=False, axis=1)
y_train = df_train[['quality']]

x_test = df_test.drop('quality', inplace=False, axis=1)
y_test = df_test[['quality']]

alpha = 0.09
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

joblib.dump(model, storing_path + '/CalificacionVinos.pkl')


