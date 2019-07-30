import PIL.Image as Image
import numpy as np
import cv2

img = cv2.imread('./debug_mask.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imwrite("debug_mask_contours.jpg", img)

# 轮廓的数量
num_contours = len(contours)

for countor in contours:
    # countor shape[num_points,2] 即[轮廓点的个数,2(x,y)]
    countor_ = np.squeeze(countor)
    print(countor_.shape)


def get_mask_contours(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #腐蚀较小的区域，特别是在预测的mask中非常常见。
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded = cv2.erode(binary, erode_kernel)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    imgname, prefix = image_path.split(".")
    save_path = imgname + "_visal." + prefix
    # cv2.imwrite(save_path, img)
    return contours
