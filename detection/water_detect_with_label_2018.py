import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm

def intersect(boxA, boxB):  # boxA is small, boxB is large
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # interArea = max(0, xB - xA) * max(0, yB - yA)
    # boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    # if interArea < 0.5 * boxA_area:
    if xB - xA < 5 or yB - yA < 5:
        return None
    return (xA, yA, xB, yB)


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



orig_jpg_path = '/home/mist/rawimages/yangon_2018/'
orig_xml_path = '/home/mist/label_data/2018_xml/'

filenames = [f[:-4] for f in os.listdir(orig_jpg_path) if f.endswith('.tif')]

h = 608
w = 608
step = 608
cnt_all = 0
cnt_water = 0
cnt_obj = 0
cnt_obj_border = 0
cnt_fn = 0  # how many objects are there in image cuts identified as no-water
cnt_fn_border = 0
    
for f in tqdm(filenames):
    orig_jpg = orig_jpg_path + f + '.tif'
    jpg = cv2.imread(orig_jpg)
    
    orig_xml = orig_xml_path + f + '.xml'
    exists_label = os.path.exists(orig_xml)
    if not exists_label: 
        continue

    fxml = open(orig_xml, 'r')
    xml = fxml.read()
    fxml.close()
    root = ET.fromstring(xml)
    xmin, ymin, xmax, ymax = [], [], [], []
    
    for item in root.findall('object'):
        xmin.append(eval(item.find('bndbox/xmin').text))
        ymin.append(eval(item.find('bndbox/ymin').text))
        xmax.append(eval(item.find('bndbox/xmax').text))
        ymax.append(eval(item.find('bndbox/ymax').text))
        
    xrange = []
    yrange = []
    
    p = 0
    while p < jpg.shape[1]:
        if p+w < jpg.shape[1]:
            xrange.append((p, p+w))
        else:
            xrange.append((jpg.shape[1]-w, jpg.shape[1]))
            break
        p += step
        
    p = 0
    while p < jpg.shape[0]:
        if p+h < jpg.shape[0]:
            yrange.append((p, p+h))
        else:
            yrange.append((jpg.shape[0]-h, jpg.shape[0]))
            break
        p += step

    for i in range(len(xrange)):
        for j in range(len(yrange)):
            cnt_all += 1
            imin, imax, jmin, jmax = xrange[i][0], xrange[i][1], yrange[j][0], yrange[j][1]
            img = jpg[jmin:jmax, imin:imax, :]
            if get_water(img) is None:  # no water, possible false negative
                for k in range(len(xmin)):
                    box1 = (xmin[k], ymin[k], xmax[k], ymax[k])
                    box2 = (imin, jmin, imax, jmax)
                    inter = intersect(box1, box2)
                    if inter:
                        cnt_obj += 1
                        cnt_fn += 1
                        if not (xmin[k] >= imin and xmax[k] < imax and ymin[k] >= jmin and ymax[k] < jmax):
                            cnt_fn_border += 1
                            cnt_obj_border += 1
            else:
                cnt_water += 1
                for k in range(len(xmin)):
                    box1 = (xmin[k], ymin[k], xmax[k], ymax[k])
                    box2 = (imin, jmin, imax, jmax)
                    inter = intersect(box1, box2)
                    if inter:
                        cnt_obj += 1
                        if not (xmin[k] >= imin and xmax[k] < imax and ymin[k] >= jmin and ymax[k] < jmax):
                            cnt_obj_border += 1

print('number of image cuts:', cnt_all)
print('number of image cuts with water', cnt_water)
print('number of objects', cnt_obj)
print('number of objects on border:',cnt_obj_border)
print('number of omitted objects:',cnt_fn)
print('number of omitted objects on border:',cnt_fn_border)
