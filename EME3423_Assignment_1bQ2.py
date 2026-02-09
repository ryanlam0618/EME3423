import cv2
import numpy as np

# Open camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, image = capture.read()
    if not ret:
        break
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert to grayscale for Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray, 100, 200)
    
    # Display all 4 windows
    cv2.imshow('Original', image)
    cv2.imshow('Gaussian Blur', gaussian_blur)
    cv2.imshow('HSV Color Space', hsv)
    cv2.imshow('Canny Edges', canny_edges)
    
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()