import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('rice.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#ACT 1: Start up
ret, thresh1 = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)

cv2.imshow('A1- Ana_gorsel', image)
cv2.imshow('A1- Esikleme', thresh1)

cv2.waitKey()


#ACT 2: Karartma işlemi
gamma = 1.8
gamma_corrected = np.array(255 * (image/255) ** gamma, dtype='uint8')
img2 = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img2, 140, 255, cv2.THRESH_BINARY)

cv2.imshow("A2- Gamma Donusum", gamma_corrected)
cv2.imshow('A2- Gammali gri', img2)
cv2.imshow('A2- Gammali tresh', thresh2)
cv2.waitKey()

#ACT 3: Dedection algoritm

blur = cv2.GaussianBlur(img, (11, 11), 0)
plt.imshow(blur, cmap='gray')
plt.show()

canny = cv2.Canny(blur, 30, 150, 3)
plt.imshow(canny, cmap='gray')
plt.show()

dilated = cv2.dilate(canny, (1, 1), iterations=0)
plt.imshow(dilated, cmap='gray')
plt.show()




(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

plt.imshow(rgb)


print("Rices in the image : ", len(cnt))