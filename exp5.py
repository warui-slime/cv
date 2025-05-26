import cv2
import numpy as np

# Load image in grayscale

img = cv2.imread('orignal.png', cv2.IMREAD_GRAYSCALE)


# Convert to float32
gray = np.float32(img)

# Apply Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)  # Dilate to mark corners

# Convert grayscale image to color for marking
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Mark corners in red
img_color[dst > 0.01 * dst.max()] = [0, 0, 255]

# Save the output image
cv2.imwrite('harris_corners_output.png', img_color)
