import requests
import getopt
import sys
import json

url = "http://10.30.15.37:5001/invocations"

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
            fixed_acidity = float(current_value)
        if current_argument in ("-fa", "--volatile_acidity"):
            volatile_acidity = float(current_value)
        if current_argument in ("-fa", "--citric_acid"):
            citric_acid = float(current_value)
        if current_argument in ("-fa", "--residual_sugar"):
            residual_sugar = float(current_value)
        if current_argument in ("-fa", "--chlorides"):
            chlorides = float(current_value)
        if current_argument in ("-fa", "--free_sulfur_dioxide"):
            free_sulfur_dioxide = float(current_value)
        if current_argument in ("-fa", "--total_sulfur_dioxide"):
            total_sulfur_dioxide = float(current_value)
        if current_argument in ("-fa", "--density"):
            density = float(current_value)
        if current_argument in ("-fa", "--ph"):
            ph = float(current_value)
        if current_argument in ("-fa", "--sulphates"):
            sulphates = float(current_value)
        if current_argument in ("-fa", "--alcohol"):
            alcohol = float(current_value)

    inference_request = {
        "columns": [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ],
        "data": [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]]
    }

    response = requests.post(url, json=inference_request)
    return(json.loads(response.text))

print(predict())
