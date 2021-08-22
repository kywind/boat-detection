import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm


def extract_image_gradient(src_image):
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3)
    y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3)
    gradient_image = np.hypot(x_gradient, y_gradient)
    ret, gradient_image = cv2.threshold(gradient_image, 40, 1000, cv2.THRESH_BINARY)
    return gradient_image,src_image


def get_water(img):
    thres_min=100
    thres_max=500
    tmp_image, s2 = extract_image_gradient(img)
    
    kernel = np.ones((1,1), np.uint8) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel) 
    erosion = cv2.erode(tmp_image, kernel) 
    tmp_image = cv2.dilate(erosion, kernel)

    gray_img = np.clip(tmp_image, 0, 255)
    gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    gray_img = np.uint8(gray_img)
    
    contours, hierarchy = cv2.findContours(gray_img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key = cv2.contourArea, reverse=True)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        if w==608 or h==608:
            continue
        if w*h < thres_min*thres_min or w*h > thres_max*thres_max:
            continue
        # img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        # gray_img = cv2.rectangle(gray_img, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.imwrite(vis_path+fn+'_gray.jpg', gray_img)
        return img
    return None


orig_imglist = 'imglist/testval.txt'
nowater_imglist = 'imglist/testval_nowater.txt'
haswater_imglist = 'imglist/testval_haswater.txt'

fin = open(orig_imglist)
data = fin.read().strip().split('\n')
fin.close()
f_no = open(nowater_imglist, 'w')
f_has = open(haswater_imglist, 'w')
for f in tqdm(data):
    img = cv2.imread(f)
    if get_water(img) is None:
        f_no.write('{}\n'.format(f))
    else:
        f_has.write('{}\n'.format(f))
f_no.close()
f_has.close()