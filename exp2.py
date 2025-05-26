import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('orignal.png')


#Averaging kernel (3x3)
kernel = np.ones((3, 3), np.float32) / 9
# kernel = np.full((3,3),5,np.float32)/27
#Applying filter
smoothed = cv2.filter2D(image, -1, kernel)
cv2.imwrite('smoothed.png',smoothed)



image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Adding Gaussian noise
noise_level = 10
mean = 0
stddev = np.sqrt(noise_level / 100.0)
gaussian_noise = np.random.normal(mean, stddev, image.shape)
noisy_image = image + gaussian_noise * 255
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Create averaging kernel
mask_size = 3
kernel = np.ones((mask_size, mask_size), np.float32) / (mask_size * mask_size)

# Applying filter
filtered_image = cv2.filter2D(noisy_image, -1, kernel)
cv2.imwrite('noisy_image.png', noisy_image)
cv2.imwrite('filtered_image.png', filtered_image)
