import pandas as pd
import sys
import getopt
import joblib


def predict():
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    short_options = "f:v:c:r:c:s:t:d:p:z:a:"
    long_options = ["fixed_acidity=", "volatile_acidity=", "citric_acid=", "residual_sugar=", "chlorides=", "free_sulfur_dioxide=", "total_sulfur_dioxide=", "density=", "ph=", "sulphates=", "alcohol="]
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-fa", "--fixed_acidity"):
            fixed_acidity = current_value
        if current_argument in ("-fa", "--volatile_acidity"):
            volatile_acidity = current_value
        if current_argument in ("-fa", "--citric_acid"):
            citric_acid = current_value
        if current_argument in ("-fa", "--residual_sugar"):
            residual_sugar = current_value
        if current_argument in ("-fa", "--chlorides"):
            chlorides = current_value
        if current_argument in ("-fa", "--free_sulfur_dioxide"):
            free_sulfur_dioxide = current_value
        if current_argument in ("-fa", "--total_sulfur_dioxide"):
            total_sulfur_dioxide = current_value
        if current_argument in ("-fa", "--density"):
            density = current_value
        if current_argument in ("-fa", "--ph"):
            ph = current_value
        if current_argument in ("-fa", "--sulphates"):
            sulphates = current_value
        if current_argument in ("-fa", "--alcohol"):
            alcohol = current_value

    dict = {
        'fixed acidity':fixed_acidity,
        'volatile acidity':volatile_acidity,
        'citric acid':citric_acid,
        'residual sugar':residual_sugar,
        'chlorides':chlorides,
        'free sulfur dioxide':free_sulfur_dioxide,
        'total sulfur dioxide':total_sulfur_dioxide,
        'density':density,
        'pH':ph,
        'sulphates':sulphates,
        'alcohol':alcohol
    }

    x_test = pd.DataFrame([dict])

    model = joblib.load('./modelos/CalificacionVinos.pkl')

    print(type(model))

    return (model.predict(x_test))

print(predict())