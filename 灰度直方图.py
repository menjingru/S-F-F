import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("/home/deepliver2/Disksdb/menjingru/R-C.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image, "gray")
plt.show()
plt.hist(image.ravel(), 256,color='black')
plt.show()
ret1, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(th1, "gray")
plt.show()

