import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('orignal.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite('output.webp', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_WEBP_QUALITY), 90])
cv2.imwrite('gray.png',gray)
equalized = cv2.equalizeHist(gray)
cv2.imwrite('equalized.png',equalized)


kernel = np.array([[ 0, -1,  0],
                    [-1, 8, -1],
                    [ 0, -1,  0]])
sharpened = cv2.filter2D(image,-1,kernel)
cv2.imwrite('sharpened.png',sharpened)
