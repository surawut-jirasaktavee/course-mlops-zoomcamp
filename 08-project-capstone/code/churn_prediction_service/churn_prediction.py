import pickle
from flask import Flask, request, jsonify


app = Flask('customer_churn_prediction')

with open('model.pkl', 'rb') as f_in:
    (model, dv) = pickle.load(f_in)

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
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)