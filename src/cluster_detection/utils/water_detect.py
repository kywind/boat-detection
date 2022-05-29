import cv2
import os
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import shutil
from scipy.optimize import curve_fit


image_file_path = './images/test/'
out_path = './images/test_has_water/'
vis_path = './images/vis_test_has_water/'
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
os.makedirs(vis_path, exist_ok=True)
thres_min=100
thres_max=200
#加载原始图像
def load_image(image_file_path):
 
    if not os.path.exists(image_file_path):
        print("图像文件不存在！")
        #sys.exit()
    else:
        img = cv2.imread(image_file_path)
        if img is None:
            print('读取图像失败!')
            #sys.exit()
        else:
            return img

#提取图像梯度信息
def extract_image_gradient(src_image):
    #转灰度图像
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
 
    #Sobel算子提取图像梯度信息
    x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3)
    y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3)
 
    #计算梯度幅值
    gradient_image = np.hypot(x_gradient, y_gradient)
    ret, gradient_image = cv2.threshold(gradient_image, 40, 1000, cv2.THRESH_BINARY)
    #gradient_image = np.uint8(np.sqrt(np.multiply(x_gradient,x_gradient)+ np.multiply(y_gradient,y_gradient)))
    
    return gradient_image,src_image

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area
 
def batch_detect(image_dir):
    img_filelist = os.listdir(image_dir)
    print('开始批量提取水域')
    i = 1
    for img_file in img_filelist:
        if img_file.find('.jpg') < 0:
            continue
        #print(image_dir + img_file)
        src_img = load_image(image_dir + img_file)
        cp_src = src_img
        #print(type(src_img))
        #src_img = cv2.imread(src_img)
        tmp_image,s2 = extract_image_gradient(src_img)
        kernel=np.ones((1,1), np.uint8) #设置卷积核
        #通过腐蚀膨胀尽量消除躁点
        erosion=cv2.erode(tmp_image, kernel) 
        tmp_image=cv2.dilate(erosion, kernel) 
        erosion=cv2.erode(tmp_image, kernel) 
        tmp_image=cv2.dilate(erosion, kernel) 
        erosion=cv2.erode(tmp_image, kernel) 
        tmp_image=cv2.dilate(erosion, kernel) 
        erosion=cv2.erode(tmp_image, kernel) 
        tmp_image=cv2.dilate(erosion, kernel) 
        cv2.imwrite(vis_path+'gradient_'+img_file, tmp_image)
        gray_img = cv2.imread(vis_path+'gradient_'+img_file, 0)
        #gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray_img,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 画出水域区域（粗糙）
        contours.sort(key = cnt_area, reverse=True)
        src=src_img
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        cnt=0
        for bbox in bounding_boxes:
            if cnt>5:
                break
            [x, y, w, h] = bbox
            if w==512 or h==512:
                continue
            if w*h < thres_min*thres_min or w*h >thres_max*thres_max:
                continue
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cnt+=1
        i += 1
        if i%500 == 0:
            print('已完成第',i,'张')
        if cnt == 0:
            continue
        shutil.copy(image_file_path+img_file, out_path+img_file)
        cv2.imwrite(vis_path+'wat_'+img_file, src)  
    print('处理完成')

if __name__ == '__main__':
    batch_detect(image_file_path)

