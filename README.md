# sign_detection project using MediaPipe

## Overview

This project detects hand signs using computer vision and MediaPipe's Hand Landmarker. It detects multiple signs such as:

- Open hand  
- Surfing sign  
- Peace sign  
- Fist

## Features

- Real-time hand sign detection from webcam video stream  
- Visualization of detected hand landmarks and recognized signs  
- REST API endpoint (`/predict`) that accepts images in JSON format for sign detection  
- Benchmark on a dataset of 10 images  

---

## Installation

### Prerequisites

- `Python`
- `mediapipe`  
- `opencv-python`  
- `fastapi`  
- `uvicorn`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the model file
```
import os
os.system('wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
```
---

## Run the project

### Run Real-Time Webcam Sign Detection

```bash
python main.py
```
This will open your webcam feed.

The program detects hands and classifies signs such as Open Hand, Peace, Surfing, and Fist.

Detected hand landmarks and the recognized sign label will be drawn on the video display.

### Run API server

```bash
uvicorn api:app --reload
```


This launches a server at `http://127.0.0.1:8000`

The `/predict` endpoint accepts POST requests with a JSON payload containing a base64-encoded hand image.

### Benchamrking

The `benchmark.py `script is used to test the accuracy and reliability of the `/predict` API endpoint by running it on a batch of static hand sign images.

This script automates the process of:

- Reading multiple image files from the images/ directory.

- Encoding each image into base64 format.

- Sending the encoded image to the /predict API.

- Displaying the predicted sign result for each image.
## Run benchmark

```bash
python benchmark.py
```