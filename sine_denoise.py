import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image and convert it to grayscale
image = cv2.imread('images/Chessboard_00451_2.png', 0)

# Compute the 2D Fourier transform of the image
fourier_transform = np.fft.fft2(image)

# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier_transform)

# Create a mask with two lines
radius = 10
mask1 = np.ones_like(image)*255
mask2 = np.zeros_like(image)

center_y = mask1.shape[0] // 2
center_x = mask1.shape[1] // 2

cv2.line(mask2, (center_x+8,center_y),(center_x+8,center_y), (255,255,255), 1)[0]
cv2.line(mask2, (center_x-8,center_y),(center_x-8,center_y), (255,255,255), 1)[0]
mask = mask1 - mask2

# Apply the mask to the shifted Fourier transform
dft_shift_masked = np.multiply(fourier_shift,mask) / 255

# Shift the origin back to the upper left corner
back_ishift_masked = np.fft.ifftshift(dft_shift_masked)

# Compute the inverse Fourier transform to obtain the filtered image
img_filtered = np.fft.ifft2(back_ishift_masked)

# Clip and convert the image back to uint8 format
img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)

# Apply a Gaussian filter to smooth the image
blur = cv2.GaussianBlur(img_filtered,(11,11),0)
blur_img = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

# Display the original and filtered images side by side
plt.subplot(1,2,1)
plt.imshow(image, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(blur_img, cmap = 'gray')
plt.show()