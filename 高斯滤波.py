import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np
from skimage.io import imread
import cv2
import scipy.fftpack as fp
from PIL import Image


# 加载图像
image_path = r"img.jpg"
image = cv2.imread(image_path)

# 将图像转换为RGB（OpenCV读取的图像为BGR）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 对图像添加高斯噪声
def add_gauss_noise(image, mean=0, val=0.01):
    size = image.shape
    # 对图像归一化处理
    image = image / 255.0
    gauss = np.random.normal(mean, val**0.05, size)
    image = image + gauss
    return image

noisy_image = add_gauss_noise(image_rgb)

# 高斯滤波
denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
def plt_def():
    # 显示原始图像、带噪声的图像和去噪图像
    plt.figure(figsize=(15, 30))

    plt.subplot(131)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(noisy_image)
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(denoised_image)
    plt.title('Denoised Image')
    plt.axis('off')

    plt.show()
plt_def()
