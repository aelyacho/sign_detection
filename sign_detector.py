'''
This module contains the function to detect hand signs based on landmarks.
It uses the landmarks of the hand to determine which sign is being made.
More info: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=en

Note: This function is made for use with a right hand and the hand palm facing the camera
The predefined signs are:
- Peace Sign
- Surfing Sign
- Open Hand
- Closed Fist 
'''

def detect_sign(landmarks):
    '''Detects hand signs based on 21 hand landmarks 
    The landmarks are expected to be in the format provided by MediaPipe HandLandmarker.
    The function checks the positions of specific landmarks to determine the sign being made.
    '''
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_folded = thumb_tip.x > landmarks[2].x 
    index_up = index_tip.y < landmarks[6].y
    middle_up = middle_tip.y < landmarks[10].y
    ring_up = ring_tip.y < landmarks[14].y
    pinky_up = pinky_tip.y < landmarks[18].y

    # Peace sign
    if index_up and middle_up and not ring_up and not pinky_up and thumb_folded:
        return "Peace Sign"
    # Surfing sign
    elif not thumb_folded and pinky_up and not index_up and not middle_up and not ring_up:
        return "Surfing Sign"
    # Open hand
    elif index_up and middle_up and ring_up and pinky_up and not thumb_folded:
        return "Open Hand"
    # Closed fist
    elif not index_up and not middle_up and not ring_up and not pinky_up and thumb_folded:
        return "Closed Fist"
    # # Thumbs up
    # elif not thumb_folded and not index_up and not middle_up and not ring_up and not pinky_up:
    #     return "Thumbs Up"
    else:
        return "Unknown"
