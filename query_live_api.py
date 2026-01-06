import requests

API_URL = "https://uda-mlops-4-6.onrender.com/inference"

# Example census data
data = {
    "age": 40,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# POST request to live API
response = requests.post(API_URL, json=data)

# Print results
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
print(f"Prediction: {response.json()['predictions'][0]}")
