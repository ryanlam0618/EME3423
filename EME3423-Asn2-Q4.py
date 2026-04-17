# EME3423 Assignment 2 - Question 4
# Exercise Rep Counter using MediaPipe Pose Detection
# Counts bicep curls with correct form
# Shows rep count and stage (up/down) on the window

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point 1
    b = np.array(b)  # Point 2 (middle joint)
    c = np.array(c)  # Point 3

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Setup camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Rep counter variables
counter = 0
stage = None
feedback = "Get Ready!"

## Setup MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while cam.isOpened():
        ret, img = cam.read()

        if not ret:
            print("[ERROR] Failed to capture frame!")
            break

        # Recolor image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        
        # Make detection
        results = pose.process(imgRGB)
        
        # Recolor back to BGR
        imgRGB.flags.writeable = True
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for LEFT arm (shoulder, elbow, wrist)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Also get RIGHT arm for more robust detection
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles for both arms
            left_angle = calculate_angle(shoulder, elbow, wrist)
            right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            # Use the arm with smaller angle (more bent = active arm)
            angle = min(left_angle, right_angle)

            # Display angle on screen
            cv2.putText(img, f"Angle: {angle:.1f}",
                        tuple(np.multiply(elbow, [1280, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            # Stage definitions:
            # - "up" = arm fully extended (angle > 160)
            # - "down" = arm fully bent (angle < 50)
            if angle > 160:
                stage = "UP"
                feedback = "Curl Down!"
            if angle < 50 and stage == 'UP':
                stage = "DOWN"
                counter += 1
                feedback = "Good! Curl Up!"

        except:
            pass

        # ========== Display Rep Counter at BOTTOM LEFT ==========
        # Setup status box
        cv2.rectangle(img, (0, 0), (320, 130), (50, 50, 50), -1)
        cv2.rectangle(img, (0, 0), (320, 130), (0, 255, 0), 3)

        # Title
        cv2.putText(img, 'BICEP CURL COUNTER', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Rep count
        cv2.putText(img, 'REPS:', (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(counter), (120, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # Stage
        cv2.putText(img, 'STAGE:', (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(stage) if stage else "---", (130, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # ========== Display Feedback at TOP CENTER ==========
        if feedback:
            cv2.rectangle(img, (400, 10), (880, 60), (0, 0, 0), -1)
            cv2.putText(img, feedback,
                        (420, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # ========== Display Instructions at TOP LEFT ==========
        cv2.putText(img, "Instructions:", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "- Curl arm when angle < 50", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "- Extend arm when angle > 160", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "- SPACE to reset counter", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "- ESC to exit", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # ========== Render Pose Landmarks ==========
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=3)
                                  )

        cv2.imshow('Exercise Rep Counter - Q4', img)

        # Press SPACE to reset counter
        if cv2.waitKey(5) & 0xFF == 32:
            counter = 0
            feedback = "Counter Reset!"

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()
print("[INFO] Exercise counter closed. Total reps:", counter)
