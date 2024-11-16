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

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # 添加盐噪声
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = np.random.choice(range(image.size), size=int(num_salt), replace=False)
    noisy_image.flat[salt_coords] = 255

    # 添加椒噪声
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = np.random.choice(range(image.size), size=int(num_pepper), replace=False)
    noisy_image.flat[pepper_coords] = 0

    return noisy_image



# 添加椒盐噪声
salt_prob = 0.1  # 盐噪声概率
pepper_prob = 0.1  # 椒噪声概率
noisy_image = add_salt_and_pepper_noise(image_rgb, salt_prob, pepper_prob)

# 使用中值滤波去噪
denoised_image = cv2.medianBlur(noisy_image, 3)
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
