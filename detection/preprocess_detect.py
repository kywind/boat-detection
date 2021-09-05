import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm
from water_detect import get_water

# vis_path = 'detect_buffer_vis/'
# if os.path.exists(vis_path):
#     shutil.rmtree(vis_path)
# os.mkdir(vis_path)


years = (2010,)
for year in years:
    orig_jpg_path = '/home/mist/rawimages/yangon_{}/'.format(year)
    new_jpg_path = 'detect_buffer_jpg_{}/'.format(year)
    new_jpg_path_nowater = 'detect_buffer_jpg_{}_nowater/'.format(year)
    # orig_xml_path = '/mnt/2018/2018_xml_in_cluster/'
    # new_xml_path = 'detect_buffer_annotation/'
    
    if os.path.exists(new_jpg_path):
        shutil.rmtree(new_jpg_path)
    os.mkdir(new_jpg_path)
    if os.path.exists(new_jpg_path_nowater):
        shutil.rmtree(new_jpg_path_nowater)
    os.mkdir(new_jpg_path_nowater)
    
    # if year == 2018:
    #     if os.path.exists(new_xml_path):
    #         shutil.rmtree(new_xml_path)
    #     os.mkdir(new_xml_path)

    filenames = [f[:-4] for f in os.listdir(orig_jpg_path) if f.endswith('.tif')]
    random.shuffle(filenames)
    # has_label = (year == 2018)
    water_detect = True

    h = 608
    w = 608
    step = 512
    cnt = 0
    cnt_all = 0
    start_time = time.time()
    
    for f in tqdm(filenames):
        orig_jpg = orig_jpg_path + f + '.tif'
        jpg = cv2.imread(orig_jpg)
        
        # orig_xml = orig_xml_path + f + '.xml'
        # exists_label = os.path.exists(orig_xml)
        # if has_label and exists_label:
        #     fxml = open(orig_xml, 'r')
        #     xml = fxml.read()
        #     root = ET.fromstring(xml)
        #     xmin, ymin, xmax, ymax = [], [], [], []
        #     
        #     for item in root.findall('object'):
        #         xmin.append(eval(item.find('bndbox/xmin').text))
        #         ymin.append(eval(item.find('bndbox/ymin').text))
        #         xmax.append(eval(item.find('bndbox/xmax').text))
        #         ymax.append(eval(item.find('bndbox/ymax').text))
        #     fxml.close()
            
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
                if water_detect:
                    flag = get_water(img)
                    
                if flag:
                    cnt += 1
                    fn = '{}{}_{}_{}.jpg'.format(new_jpg_path,f,imin,jmin)
                    cv2.imwrite(fn, img)
                
                else:
                    fn = '{}{}_{}_{}.jpg'.format(new_jpg_path_nowater,f,imin,jmin)
                    cv2.imwrite(fn, img)
                    
                    # if has_label and exists_label:         
                    #     data = ET.Element('annotation')
                    #
                    #     for k in range(len(xmin)):
                    #         if xmin[k] >= imin and xmax[k] < imax and ymin[k] >= jmin and ymax[k] < jmax:
                    #             obj = ET.SubElement(data, 'object')
                    #             box = ET.SubElement(obj, 'bndbox')
                    #             x1 = ET.SubElement(box, 'xmin')
                    #             y1 = ET.SubElement(box, 'ymin')
                    #             x2 = ET.SubElement(box, 'xmax')
                    #             y2 = ET.SubElement(box, 'ymax')
                    #             x1.text = str(xmin[k]-imin)
                    #             x2.text = str(xmax[k]-imin)
                    #             y1.text = str(ymin[k]-jmin)
                    #             y2.text = str(ymax[k]-jmin)
                    #             
                    #     mydata = ET.tostring(data)
                    #     fout = open('{}{}_{}_{}.xml'.format(new_xml_path,f,imin,jmin), 'wb+')
                    #     fout.write(mydata)
                    #     fout.close()
                    
        # if cnt >= 500:
        #     break
    
    end_time = time.time()
    print('year {} finished.'.format(year))
    print('all:', cnt_all)
    print('has water:',cnt)
    print('time:', end_time-start_time)
