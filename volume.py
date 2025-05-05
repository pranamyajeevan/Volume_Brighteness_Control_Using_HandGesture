import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import screen_brightness_control as abc

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)#Confidence threshold for detecting a hand.
mpDraw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# Initialize variables
volbar = 400
volper = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand gestures
    results = hands.process(imgRGB)

    # Separate landmarks for left and right hands
    left_hand_lm = []
    right_hand_lm = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            lmList = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if label == "Left":
                left_hand_lm = lmList
            elif label == "Right":
                right_hand_lm = lmList

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Volume control with the right hand (thumb and index finger distance)
    if right_hand_lm:
        x1, y1 = right_hand_lm[4][1], right_hand_lm[4][2]  # Thumb tip
        x2, y2 = right_hand_lm[8][1], right_hand_lm[8][2]  # Index finger tip
        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [30, 350], [volMin, volMax])
        volbar = np.interp(length, [30, 350], [400, 110])
        volper = np.interp(length, [30, 350], [0, 100])

        print(f"Volume: {vol}, Length: {int(length)}")
        volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Brightness control with the left hand (thumb and index finger distance)
    if left_hand_lm:
        x3, y3 = left_hand_lm[4][1], left_hand_lm[4][2]  # Thumb tip
        x4, y4 = left_hand_lm[8][1], left_hand_lm[8][2]  # index finger tip
        cv2.circle(img, (x3, y3), 13, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x4, y4), 13, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 3)

        length_brightness = hypot(x4 - x3, y4 - y3)
        bright = np.interp(length_brightness, [15, 220], [0, 100])

        print(f"Brightness: {bright}, Length: {length_brightness}")
        abc.set_brightness(int(bright))

    # Display the video feed
    cv2.imshow('Hand Gesture Control', img)

    # Exit the loop when 'k' is pressed
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()