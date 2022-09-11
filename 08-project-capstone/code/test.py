import requests


object = {
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
response = requests.post(URL, json=object).json()
print(response)
