import cv2
import numpy as np


def sine_denoise(image):
    # Compute the 2D Fourier transform of the image
    fourier_transform = np.fft.fft2(image)
    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier_transform)
    # Create a mask with two lines
    mask1 = np.ones_like(image) * 255
    mask2 = np.zeros_like(image)
    center_y = mask1.shape[0] // 2
    center_x = mask1.shape[1] // 2
    var = cv2.line(mask2, (center_x + 8, center_y), (center_x + 8, center_y), (255, 255, 255), 1)[0]
    var2 = cv2.line(mask2, (center_x - 8, center_y), (center_x - 8, center_y), (255, 255, 255), 1)[0]
    mask = mask1 - mask2
    # Apply the mask to the shifted Fourier transform
    dft_shift_masked = np.multiply(fourier_shift, mask) / 255
    # Shift the origin back to the upper left corner
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    # Compute the inverse Fourier transform to obtain the filtered image
    img_filtered = np.fft.ifft2(back_ishift_masked)
    # Clip and convert the image back to uint8 format
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)
    return img_filtered


def process(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(sine_denoise(gray), (5, 5), 0)
    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(img_blur, 30, 55, apertureSize=3)
    # Close the image with Dilation and Erosion to connect fractured lines
    kernel = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=2)
    ero = cv2.erode(dilate, kernel, iterations=1)
    # Calculate Black and White pixels to detect if the original image  has salt and pepper noise
    number_of_white_pix = np.sum(ero == 255)
    number_of_black_pix = np.sum(ero == 0)
    if number_of_black_pix < number_of_white_pix:
        noiseless_image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
        noiseless_gray = cv2.cvtColor(noiseless_image, cv2.COLOR_BGR2GRAY)
        noiseless_blur = cv2.medianBlur(noiseless_gray, 3)
        noiseless_edges = cv2.Canny(noiseless_blur, 30, 55, apertureSize=3)
        noiseless_dilate = cv2.dilate(noiseless_edges, kernel, iterations=2)
        noiseless_ero = cv2.erode(noiseless_dilate, kernel, iterations=1)
        cv2.imshow("Test2", noiseless_ero)
        return noiseless_ero
    cv2.imshow("Test2", ero)
    return ero


# Load and process the image
img = cv2.imread('images/Chessboard_0451.png') 
processed = process(img)

# Apply the Hough transform to find lines in the image
lines = cv2.HoughLines(processed, 1, np.pi / 180, 180)

filtered_lines = []
check = []
check_horizon = []
check_vertical = []
i = 0

for line in lines:
    exist = False
    rho, theta = line[0]
    # Convert line offset and angle to positive integer and  degree
    rho_c = abs(int(rho))
    theta_c = int(theta / np.pi * 180)
    if 150 <= theta_c <= 210:
        theta_c = abs(theta_c - 180)
    # Skip diagonal lines
    if (30 < theta_c < 75) or (105 < theta_c < 150):
        i += 1
        exist = True
    # Catch first line detected since it is always correct
    if i == 0:
        check.append((theta_c, rho_c))
        if check[-1][0] > 45:
            check_horizon.append(check[-1])
        else:
            check_vertical.append(check[-1])
        filtered_lines.append(lines[i])
        i += 1
    else:
        # Remove lines at the same position (Offset and Angle similar to each other)
        for check_line in check:
            if abs(theta_c - check_line[0]) <= 3 and abs(rho_c - check_line[1]) <= 16:
                i += 1
                exist = True
                break
        # Detect if line is vertical or horizontal
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

print(f'The chessboard size is {rows}x{cols} ({rows} rows and {cols} columns)')
rows_del = abs(rows - 8)
cols_del = abs(cols - 12)
accuracy = 100 - (rows_del/8 + cols_del/12)/2 * 100
print(f'Accuracy: {accuracy}%')

# Draw lines on the image to show the grid of the chessboard
i = 0
# Red, Orange, Yellow, Green, Blue
rgb = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]

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
    i += 1
    if i == len(rgb):
        i = 0

# Display the image with lines drawn on it
cv2.imshow('Image with lines', img)
cv2.waitKey(0)
