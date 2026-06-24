import requests

payload = {"features": [0.1] * 30}
response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
print(response.status_code)
print(response.json())
