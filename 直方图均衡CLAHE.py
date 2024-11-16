import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(grayHist):
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的范围
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()

if __name__ == "__main__":
    # 读取图像并转换为灰度图
    img = cv2.imread(r'image.jpg', 0)
    # 图像的灰度级范围是0~255
    grayHist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plot(grayHist)

"""
    equ = cv2.equalizeHist(img)  # 输入为灰度图
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imwrite('HE2.png', equ)

    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    res = np.hstack((img, cl1))
    cv2.imwrite('CLAHE2.jpg', cl1)
"""

