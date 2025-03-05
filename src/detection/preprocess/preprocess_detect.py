import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm
from water_detect import get_water

MAP_PATH_DICT = {
    2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20101231/',
    2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20111231/',
    2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20121231/',
    2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20131231/',
    2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20141231/',
    2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20151231/',
    2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20161231/',
    2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20171231/',
    2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20181231/',
    2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20191231/',
    2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20201231/',
    2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/2021_new/',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20230827/'
}
BUFFER_PATH_DICT = {
    2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2010/',
    2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2011/',
    2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2012/',
    2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2013/',
    2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2014/',
    2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2015/',
    2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2016/',
    2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2017/',
    2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2018/',
    2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2019/',
    2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2020/',
    2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2021/',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2023/'
}
BUFFER_NOWATER_PATH_DICT = {
    2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2010_nowater/',
    2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2011_nowater/',
    2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2012_nowater/',
    2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2013_nowater/',
    2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2014_nowater/',
    2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2015_nowater/',
    2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2016_nowater/',
    2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2017_nowater/',
    2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2018_nowater/',
    2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2019_nowater/',
    2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2020_nowater/',
    2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2021_nowater/',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2023_nowater/'
}


years = MAP_PATH_DICT.keys()
for year in years:
    orig_jpg_path = MAP_PATH_DICT[year]
    new_jpg_path = BUFFER_PATH_DICT[year]
    new_jpg_path_nowater = BUFFER_NOWATER_PATH_DICT[year]
    
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
    
    tbar = tqdm(filenames)
    for f in tbar:
        tbar.set_description('cnt_water: {}, cnt_all: {}'.format(cnt, cnt_all))
        orig_jpg = orig_jpg_path + f + '.tif'
        try: 
            jpg = cv2.imread(orig_jpg)
            tmp = jpg.shape
        except: 
            continue
        
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
