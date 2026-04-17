import cv2
import numpy as np

def nothing():
    pass

frame = np.zeros((480, 640, 3), np.uint8)  # Load an image from file

cv2.namedWindow('Trackbars')
cv2.createTrackbar('RED', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('GREEN', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('BLUE', 'Trackbars', 0, 255, nothing)
while True:
    red = cv2.getTrackbarPos('RED', 'Trackbars')
    green = cv2.getTrackbarPos('GREEN', 'Trackbars')
    blue = cv2.getTrackbarPos('BLUE', 'Trackbars')          
    cv2.imshow('frame', frame)


    frame[:, :] = (blue, green, red)  # Update the image with the new color


    if cv2.waitKey(1) == 27:
        break