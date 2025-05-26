from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sign_detector import detect_sign
from pydantic import BaseModel
import base64
'''This script sets up a FastAPI application that receives images encoded in base64 format,
processes them to detect hand landmarks using MediaPipe,
 and identifies hand signs based on those landmarks.'''
app = FastAPI()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

class ImageRequest(BaseModel):
    image: str

@app.post("/predict")
async def predict(req: ImageRequest):
    # Decode the base64 image
    image_data = base64.b64decode(req.image)
    np_img = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image) # Detect hand landmarks    

    if result.hand_landmarks:
        sign = detect_sign(result.hand_landmarks[0])
        return {"sign": sign}
    return {"sign": "No hand detected"}
