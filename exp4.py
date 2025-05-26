import cv2
import numpy as np

# Read image and convert to RGB
img = cv2.imread("orignal.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image to a 2D array of pixels
vectorized_img = img.reshape((-1, 3)).astype(np.float32)

# Define criteria and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5  # You can change this to try different numbers of clusters
attempts = 10

_, labels, centers = cv2.kmeans(vectorized_img, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

# Convert back to uint8 and recreate segmented image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# Convert back to BGR before saving
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("segmented_image_k{}.jpg".format(K), segmented_image_bgr)
