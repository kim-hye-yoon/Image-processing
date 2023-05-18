import cv2
import numpy as np

# Load the image
image = cv2.imread('Chessboard_0481_1.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)
cv2.waitKey(0)