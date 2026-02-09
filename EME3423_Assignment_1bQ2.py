import cv2
import numpy as np

# Read the image
image = cv2.imread('C:\\Users\\Ryan\\python\\.vscode\\EME3423\\Resources\\anime.jpg')  # Replace with your image path

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

cv2.waitKey(0)
cv2.destroyAllWindows()