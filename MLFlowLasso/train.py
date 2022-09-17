from sklearn.linear_model import Ridge, Lasso
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

import os
import sys 

os.environ["MLFLOW_TRACKING_URI"] = 'http://10.30.15.37:8990'
os.environ["MLFLOW_EXPERIMENT_NAME"] = "PrediccionCalidadVinos"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

print("hello")

if __name__ == "__main__":
    df_train = pd.read_csv('../data/winequality-red_train.csv', sep = ',')  
    df_test = pd.read_csv('../data/winequality-red_test.csv', sep = ',')  

    x_train = df_train.drop('quality', inplace=False, axis=1)
    y_train = df_train[['quality']]

    x_test = df_test.drop('quality', inplace=False, axis=1)
    y_test = df_test[['quality']]

    import mlflow

    try:
        exp_id = mlflow.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"]).experiment_id
    except Exception as e:
        exp_id = mlflow.create_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        
    #Set parameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #Set run name
    run_name = 'Ordinary Least Squares'

    #Set description (Markdown format) #https://google.github.io/styleguide/docguide/style.html#lists
    description = ""

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, description=description):
        model = Lasso(alpha=alpha).fit(x_train,y_train)

        y_tmp = y_test.copy()

        y_tmp['predicted'] = model.predict(x_test)
        y_tmp.drop('predicted', inplace=False, axis=1)
        y_tmp['index'] = range(1, len(y_tmp) + 1)

        y_tmp.columns = ['Real', 'Predicho', 'Index']

        #log de parametros usados
        mlflow.log_params({'alpha': alpha})
            
        metricas = {
            'MAE': mae(y_tmp[['Real']], y_tmp[['Predicho']]),
            'MSE': mse(y_tmp[['Real']], y_tmp[['Predicho']]),
            'RMSE': mse(y_tmp[['Real']], y_tmp[['Predicho']], squared=False),
            'MAPE': mape(y_tmp[['Real']], y_tmp[['Predicho']]),
        }

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(model, "Ordinary Least Squares")

        print(metricas)




