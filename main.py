import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sign_detector import detect_sign
'''
This script captures video from the webcam, detects hand landmarks using MediaPipe, and identifies hand signs based on those landmarks.
The system detects signs like Peace Sign, Surfing Sign, Open Hand, and Closed Fist.
More info: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=en'''

# os.system('wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
#os.system('wget -q https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# initialize MediaPipe HandLandmarker model
model_path = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


vidCap = cv2.VideoCapture(0)
print("Press 'ESC' to exit the video stream.")
if not vidCap.isOpened():
    print("Error: Could not open video stream.")
    exit()


with HandLandmarker.create_from_options(options) as landmarker:
    # Main loop to capture video frames
    while vidCap.isOpened():
        ret, frame = vidCap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # flip horizontally for mirror effect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                sign = detect_sign(landmarks)
                for lm in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) # draw landmarks on the frame

                cv2.putText(frame, sign, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

vidCap.release()
cv2.destroyAllWindows()
