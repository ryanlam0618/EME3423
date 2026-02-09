import cv2
import numpy as np
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = capture.read()

    imgCanny = cv2.Canny(img, 100, 100)

    kernal = np.ones((5, 5) , np.uint8)
    imgDilation = cv2.dilate(imgCanny, kernal, iterations=1)

    cv2.imshow("img", img)
    cv2.imshow("imgCanny", imgCanny)

    if cv2.waitKey(20) & 0xff == ord('q'):
        break