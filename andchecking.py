import cv2
import numpy as np

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.6)


while True:
    _, img = cam.read()
    img  = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    cv2.imshow("frame", img)

    if cv2.waitKey(20) & 0xff == ord('q'):
        break