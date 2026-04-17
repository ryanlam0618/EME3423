import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

colorDots = mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=2)
colorConnections = mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        _, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(imgRGB)

        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=colorDots, connection_drawing_spec=colorConnections)

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=colorDots, connection_drawing_spec=colorConnections)


        if results.pose_landmarks:
            print(results.pose_landmarks.landmark)

            heght, width, channel = img.shape

            cx, cy = int(results.pose_landmarks.landmark[11].x * width), int(results.pose_landmarks.landmark[11].y * heght)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), 3)

            if cy == 150:
                print("jump")



        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

