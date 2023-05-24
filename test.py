import cv2
import numpy as np

# Load the image
img = cv2.imread('images/Chessboard_00451.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the Canny edge detector to find edges in the image
edges = cv2.Canny(gray, 10, 50)

# Apply the Hough transform to find lines in the image
lines = cv2.HoughLines(edges, 1, np.pi/180, 180)

# Initialize variables to count the number of horizontal and vertical lines
horizontal_lines = 0
vertical_lines = 0

# Iterate over the lines and count the number of horizontal and vertical lines
for line in lines:
    rho, theta = line[0]
    if np.abs(theta) < np.pi/4 or np.abs(theta-np.pi) < np.pi/4:
        vertical_lines += 1
    else:
        horizontal_lines += 1

# Calculate the number of rows and columns on the chessboard
rows = horizontal_lines + 1
cols = vertical_lines - 1

print(f'The chessboard has {rows} rows and {cols} columns')

# Draw lines on the image to show the grid of the chessboard
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# Display the image with lines drawn on it
cv2.imshow('Image with lines', img)
cv2.waitKey(0)