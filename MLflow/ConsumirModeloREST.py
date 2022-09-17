import requests

url = "http://10.30.15.37:5001/invocations"

inference_request = {
    "columns": [
        "alcohol",
        "chlorides",
        "citric acid",
        "density",
        "fixed acidity",
        "free sulfur dioxide",
        "pH",
        "residual sugar",
        "sulphates",
        "total sulfur dioxide",
        "volatile acidity",
    ],
    "data": [[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]],
}

response = requests.post(url, json=inference_request)

print(response.json())