import os
import pickle

import requests
from flask import Flask, request, jsonify

from pymongo import MongoClient


EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
MODEL = os.getenv('PROD_MODEL', './model.pkl')

with open(MODEL, 'rb') as f_in:
    (model, dv) = pickle.load(f_in)


app = Flask('customer_churn_prediction')
mongo_client = MongoClient(MONGODB_ADDRESS)
database = mongo_client.get_database("churn_prediction_service")
collection = database.get_collection("data")


def label_encoding(customer):
    if list(customer.items())[1][1] == "Yes":
        customer["International plan"] = "1"
    else:
        customer["International plan"] = "0"
    
    if list(customer.items())[2][1] == "Yes":
        customer['Voice mail plan'] = "1"
    else:
        customer['Voice mail plan'] = "0"
    return customer

def predict(model, customer):
    X = dv.transform(customer)
    print(X)
    y_pred = model.predict(X)
    return y_pred


@app.route('/predict', methods=['POST'])
def churn_prediction():

    customer = request.get_json()
    customer_encoded = label_encoding(customer)
    print(customer_encoded)
    pred = predict(model, customer_encoded)
    
    if float(pred) == 0.0:
        result = {
            'customer_churn_prediction': "Not Churn"
        }
    else:
        result = {
            'customer_churn_prediction': "Churn"
        }
    print(result)
    save_to_database(customer, float(pred))
    send_to_evidently_service(customer, float(pred))
    return jsonify(result)

def save_to_database(customer, pred):
    obj = customer.copy()
    obj['prediction'] = pred
    collection.insert_one(dict(obj))


def send_to_evidently_service(customer, pred):
    obj = customer.copy()
    obj['prediction'] = pred
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/customer_churn", json=[obj])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
