import cv2
import numpy as np

# Load the image
img = cv2.imread('Images/Chessboard_0511.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply the Canny edge detector to find edges in the image
edges = cv2.Canny(img_blur, 30, 55, apertureSize=3)

# cv2.imshow("Test", edges)

kernel = np.ones((2, 2), np.uint8)
dilate = cv2.dilate(edges, kernel, iterations=2)  # < --- Added a dilate, check link I provided
ero = cv2.erode(dilate, kernel, iterations=1)
cv2.imshow("Test2", ero)

# Apply the Hough transform to find lines in the image
lines = cv2.HoughLines(ero, 1, np.pi / 180, 180)

filtered_lines = []
check = []
check_horizon = []
check_vertical = []

i = 0
for line in lines:
    exist = False
    rho, theta = line[0]
    rho_c = abs(int(rho))
    theta_c = int(theta / np.pi * 180)
    if 150 <= theta_c <= 210:
        theta_c = abs(theta_c - 180)
    if (30 < theta_c < 75) or (105 < theta_c < 150):
        i += 1
        exist = True
    if i == 0:
        check.append((theta_c, rho_c))
        if check[-1][0] > 45:
            check_horizon.append(check[-1])
        else:
            check_vertical.append(check[-1])
        filtered_lines.append(lines[i])
        i += 1
    else:
        for check_line in check:
            if abs(theta_c - check_line[0]) <= 3 and abs(rho_c - check_line[1]) <= 16:
                i += 1
                exist = True
                break
        if not exist:
            check.append((theta_c, rho_c))
            filtered_lines.append(lines[i])
            i += 1
            if check[-1][0] > 45:
                check_horizon.append(check[-1])
            else:
                check_vertical.append(check[-1])
            check_horizon = sorted(check_horizon, key=lambda last: last[-1])
            check_vertical = sorted(check_vertical, key=lambda last: last[-1])

    if i == len(lines):
        break

# Calculate the number of rows and columns on the chessboard
rows = len(check_horizon) - 1
cols = len(check_vertical) + 1

print(f'The chessboard has {rows} rows and {cols} columns')

# Draw lines on the image to show the grid of the chessboard

i = 0
rgb = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]  # Red, Orange, Yellow, Green, Blue

for line in filtered_lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(img, (x1, y1), (x2, y2), rgb[i], 2)

    theta_c = theta / np.pi * 180
    if 150 <= theta_c <= 210:
        theta_c = abs(theta_c - 180)

    i += 1
    if i == len(rgb):
        i = 0

# Display the image with lines drawn on it
cv2.imshow('Image with lines', img)
cv2.waitKey(0)