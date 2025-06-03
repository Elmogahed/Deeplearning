import requests

url = "http://127.0.0.1:5000/predict"

data = {
  "age": 70,
  "sex": "Male",
  "dataset": "Cleveland",
  "cp": "asymptomatic",
  "trestbps": 160,
  "chol": 286,
  "fbs": False,
  "restecg": "lv hypertrophy",
  "thalch": 108,
  "exang": True,
  "oldpeak": 1.5,
  "slope": "flat",
  "ca": 3,
  "thal": "normal"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
