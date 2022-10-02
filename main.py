# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np

img_path = r"D:\UserData\Code\Python\lectureCodes\AnimalClassifiction\girl.jpg"

def show_convolution():
    img = cv2.imread(img_path, 1)
    cv2.imshow("OriginalImg", img)
    cv2.waitKey(0)
    #模糊化卷积核
    kernel1 = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
    img2 = cv2.filter2D(img, -1, kernel1)
    cv2.imshow("BlurryImg", img2)
    cv2.waitKey(0)
    #锐化后的图像
    kernel2 =  np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img3 = cv2.filter2D(img, -1, kernel2)
    cv2.imshow("SharpenImg", img3)
    cv2.waitKey(0)
    #边缘检测
    kernel3 =  np.array([[1,1,1],[1,-8,1],[1,1,1]])
    img4 = cv2.filter2D(img, -1, kernel3)
    cv2.imshow("EdgeDetect", img4)
    cv2.waitKey(0)


# show_convolution()



