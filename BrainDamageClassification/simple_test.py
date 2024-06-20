import numpy as np
import cv2
import matplotlib.pyplot as plt



image_one = cv2.imread("C:\\Users\\1\\Desktop\\MachineLearningStud\\archive\\Testing\\glioma\\Te-gl_0010.jpg")
image_two = cv2.imread("C:\\Users\\1\\Desktop\\MachineLearningStud\\archive\\Testing\\glioma\\Te-glTr_0007.jpg")

image_one = cv2.resize(image_one, (500, 500))
image_two = cv2.resize(image_two, (500, 500))
result_image = (1 - 3.98) * image_one + 3.89 * image_two

def hsv_from_bgr(image):

    result_hsv_image = np.zeros(image.shape)
    for pix_i in range(image.shape[0]):
        for pix_j in range(image.shape[1]):

            color_r = image[pix_i, pix_j, 0] / 255.0
            color_g = image[pix_i, pix_j, 0] / 255.0
            color_b = image[pix_i, pix_j, 0] / 255.0

            color_max = max((color_r, max((color_g, color_b))))
            color_min = min((color_r, max((color_g, color_b))))
            color_diff = color_max - color_min

            color_h = 0
            color_s = 0
            color_v = 0

            if (color_max == color_min):
                color_h = 0
            
            elif (color_max == color_r):
                color_h = (60 * ((color_g - color_b) / color_diff) + 360) % 360
            
            elif (color_max == color_g):
                color_h = (60 * ((color_b - color_r) / color_diff) + 120) % 360
            
            elif (color_max == color_b):
                color_h = (60 * ((color_r - color_g) / color_diff) + 240) % 360
            
            if (color_max == 0):
                color_s = 0
            
            else:
                color_s = (color_diff / color_max) * 100
            
            color_v = color_max * 100

            result_hsv_image[pix_i, pix_j, 0] = color_h
            result_hsv_image[pix_i, pix_j, 1] = color_s
            result_hsv_image[pix_i, pix_j, 2] = color_v
    
    return result_hsv_image

hsv_image_one = hsv_from_bgr(image_one)
hsv_image_two = hsv_from_bgr(image_two)

fig, axis = plt.subplots(ncols=5)

axis[0].imshow(image_one)
axis[0].grid()
axis[1].imshow(image_two)
axis[0].grid()
axis[2].imshow(hsv_image_one)
axis[2].grid()
axis[3].imshow(hsv_image_two)
axis[3].grid()
axis[4].imshow(result_image)
axis[4].grid()

plt.show()