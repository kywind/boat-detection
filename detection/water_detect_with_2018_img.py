import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm
from water_detect import get_water

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

os.makedirs('visualize_nowater/', exist_ok=True)
tbar = tqdm(filenames)
for f in tbar:
    tbar.set_description('cnt_water: {}, cnt_all: {}'.format(cnt_water, cnt_all))
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
            has_water = get_water(img)
            if has_water is None:  # no water, possible false negative
                for k in range(len(xmin)):
                    box1 = (xmin[k], ymin[k], xmax[k], ymax[k])
                    box2 = (imin, jmin, imax, jmax)
                    inter = intersect(box1, box2)
                    if inter:
                        # cv2.imwrite('visualize_nowater/{}_{}_{}_gray.jpg'.format(f, imin, jmin), gray_img)
                        # cv2.imwrite('visualize_nowater/{}_{}_{}.jpg'.format(f, imin, jmin), img)
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
