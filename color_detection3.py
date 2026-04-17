import cv2
import numpy as np
def nothing():
    pass


cv2.namedWindow('Trackbars')
cv2.createTrackbar('HUELOW', 'Trackbars', 0, 255, nothing)  
cv2.createTrackbar('HUEHIGH', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('SATLOW', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('SATHIGH', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('VALLOW', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('VALHIGH', 'Trackbars', 0, 255, nothing)


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    huelow = cv2.getTrackbarPos('HUELOW', 'Trackbars')
    huehigh = cv2.getTrackbarPos('HUEHIGH', 'Trackbars')
    satlow = cv2.getTrackbarPos('SATLOW', 'Trackbars')
    sathigh = cv2.getTrackbarPos('SATHIGH', 'Trackbars')
    vallow = cv2.getTrackbarPos('VALLOW', 'Trackbars')
    valhigh = cv2.getTrackbarPos('VALHIGH', 'Trackbars')

    FGmask = cv2.inRange(hsv, (huelow, satlow, vallow), (huehigh, sathigh, valhigh))
    cv2.imshow('FGmask', FGmask)
    
    contours, hierarchy = cv2.findContours(FGmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)