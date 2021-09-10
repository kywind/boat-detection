import cv2 
import numpy as np
import time
import os

def extract_image_gradient(src_image):
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3)
    y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3)
    gradient_image = np.hypot(x_gradient, y_gradient)
    ret, gradient_image = cv2.threshold(gradient_image, 40, 1000, cv2.THRESH_BINARY)
    return gradient_image,src_image


def get_water(img):
    thres_min = 50
    thres_max = 500
    tmp_image, s2 = extract_image_gradient(img)
    
    kernel = np.ones((1,1), np.uint8) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel)
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel)

    # cache_path = 'cache/'
    # os.makedirs(cache_path, exist_ok=True)
    # cv2.imwrite(cache_path + 'gradient.jpg', tmp_image)
    # gray_img = cv2.imread(cache_path + 'gradient.jpg', 0)
    gray_img = np.clip(tmp_image, 0, 255)
    gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    gray_img = 255 * (gray_img <= 127)
    # gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    gray_img = np.uint8(gray_img)
    gray_img = np.pad(gray_img, ((2, 2), (2, 2)), mode='minimum')
    
    contours, hierarchy = cv2.findContours(gray_img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key = cv2.contourArea, reverse=True)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # vis_path = 'visualize_water/'
    # os.makedirs(vis_path, exist_ok=True)
    # print(cv2.contourArea(contours[0]), cv2.contourArea(contours[1]), cv2.contourArea(contours[2]))

    # return None 
    for idx, (contour, bbox) in enumerate(zip(contours, bounding_boxes)):
        [x, y, w, h] = bbox
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)
        if w*h < thres_min*thres_min or w*h > thres_max*thres_max or w*h/area > 3 or area/(length*length)<0.01:
            continue
        # img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        # gray_img = cv2.rectangle(gray_img, (x,y), (x+w,y+h), (0,255,0), 2)
        # img_save = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # img_save = cv2.drawContours(img_save, contours, idx, (0,0,255), 1)  

        # fn = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        # cv2.imwrite(vis_path + fn + '.jpg', img_save)
        # cv2.imwrite(vis_path + fn + '_gray.jpg', gray_img)
        return True
    return False