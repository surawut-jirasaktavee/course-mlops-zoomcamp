import requests
import json


def integration_test():
    '''
    Test a model.pkl from production app located in "../churn_prediction_service/churn_prediction.py
    raise an AssertionError if the prediction isn't correct
    Hint: it starts by a different dockerfile located in current directory
    ''' 

    customer = {
            "Account length": 117,
            "International plan": "No",
            "Voice mail plan": "No",
            "Number vmail messages": 0,
            "Total day minutes": 0,
            "Total day calls": 97,
            "Total eve minutes": 351.6,
            "Total eve calls": 80,
            "Total night minutes": 215.8,
            "Total night calls": 90,
            "Total intl minutes": 8.7,
            "Total intl calls": 4,
            "Customer service calls": 4
            }
            
    URL = "http://localhost:9696/predict"
    actual_response = requests.post(URL, json=customer).json()
    print(f"Churn prediction result: {actual_response}")

    expected_response = {'customer_churn_prediction': "Not Churn"}
    assert expected_response == actual_response

if __name__=='__main__':
    integration_test()

