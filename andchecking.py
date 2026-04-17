import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,min_detection_confidence=0.8,min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cam.read()
    img  = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)



    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)

                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):

        break