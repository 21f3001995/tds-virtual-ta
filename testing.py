import requests

res = requests.post(
    "http://127.0.0.1:8000/api/",
    json={"question": "What is gradient descent?"}
)

print(res.json())
