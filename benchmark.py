'''
Benchmark script to test the image classification API.
This script reads images from a specified directory, encodes them in base64,
and sends them to the API for prediction. The results are printed to the console.
'''
import requests
import base64
import os

dir_path = 'images/'

for filename in os.listdir(dir_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        response = requests.post("http://127.0.0.1:8000/predict", json={"image": img_b64})
        print(f"\nResult for {filename}:", end=' ')

        try:
            print(response.json().get("sign", "No sign detected"))
        except Exception as e:
            print("Error decoding response:", e)
