import numpy as np
import cv2

# Load grayscale image
img = cv2.imread("orignal.png", cv2.IMREAD_GRAYSCALE)


# Add Different Types of Noise


# 1. Gaussian Noise
gauss_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randn(gauss_noise, 128, 20)  # mean=128, stddev=20
gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
gn_img = cv2.add(img, gauss_noise)
cv2.imwrite("gaussian_noise.png", gn_img)

# 2. Uniform Noise
uni_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randu(uni_noise, 0, 255)
uni_noise = (uni_noise * 0.5).astype(np.uint8)
un_img = cv2.add(img, uni_noise)
cv2.imwrite("uniform_noise.png", un_img)

# 3. Impulse Noise (Salt Noise)
imp_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randu(imp_noise, 0, 255)
imp_noise = cv2.threshold(imp_noise, 245, 255, cv2.THRESH_BINARY)[1]
in_img = cv2.add(img, imp_noise)
cv2.imwrite("impulse_noise.png", in_img)


# Noise Removal (Denoising)


# 1. Using Non-local Means Denoising
denoised_gn = cv2.fastNlMeansDenoising(gn_img, None, 10, 10)
cv2.imwrite("denoised_gaussian.png", denoised_gn)

denoised_un = cv2.fastNlMeansDenoising(un_img, None, 10, 10)
cv2.imwrite("denoised_uniform.png", denoised_un)

denoised_in = cv2.fastNlMeansDenoising(in_img, None, 10, 10)
cv2.imwrite("denoised_impulse.png", denoised_in)


# Median Filtering

blurred_gn = cv2.medianBlur(gn_img, 3)
blurred_un = cv2.medianBlur(un_img, 3)
blurred_in = cv2.medianBlur(in_img, 3)
cv2.imwrite("median_gaussian.png", blurred_gn)
cv2.imwrite("median_uniform.png", blurred_un)
cv2.imwrite("median_impulse.png", blurred_in)


# Gaussian Filtering

gaussf_gn = cv2.GaussianBlur(gn_img, (3, 3), 0)
gaussf_un = cv2.GaussianBlur(un_img, (3, 3), 0)
gaussf_in = cv2.GaussianBlur(in_img, (3, 3), 0)
cv2.imwrite("gaussian_filtered_gn.png", gaussf_gn)
cv2.imwrite("gaussian_filtered_un.png", gaussf_un)
cv2.imwrite("gaussian_filtered_in.png", gaussf_in)


# Save Original for Reference

cv2.imwrite("original_image.png", img)
