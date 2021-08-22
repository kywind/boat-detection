import cv2
import numpy as np
import os
import sys


def blur():
    inpath = 'result/2018_single/'
    os.makedirs('temp/', exist_ok=True)
    imglist = [f for f in os.listdir(inpath) if f.endswith('.jpg')]
    for img in imglist:
        i = img.split('.')[0]
        tmp_image = cv2.imread(inpath + img)
        # gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        # x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3)
        # y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3)
        # gradient_image = np.hypot(x_gradient, y_gradient)
        # ret, gradient_image = cv2.threshold(gradient_image, 40, 1000, cv2.THRESH_BINARY)
        # tmp_image = gray_image
        kernel = np.ones((5,5), np.uint8)
        for t in range(20):
            erosion = cv2.erode(tmp_image, kernel) 
            tmp_image = cv2.dilate(erosion, kernel) 
        cv2.imwrite('temp/{}.jpg'.format(i), tmp_image)

    
def contour():
    for i in range(50):
        img = cv2.imread('temp/{}.jpg'.format(i))
        # cv2.imwrite('orig.jpg', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('gray.jpg',gray)
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('binary.jpg',binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(0,0,255),1)   
        cv2.imwrite('contour/{}_contour.jpg'.format(i), img)
        
        
def calc():
    year = sys.argv[1]
    inpath = 'result/{}_single/'.format(year)
    imglist = [f for f in os.listdir(inpath) if f.endswith('.jpg')]
    area_total = 0
    for f in imglist:
        i = f.split('.')[0]
        img = cv2.imread(inpath + f)
        kernel = np.ones((5,5), np.uint8)
        for t in range(10):
            erosion = cv2.erode(img, kernel) 
            img = cv2.dilate(erosion, kernel) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # os.makedirs('contour/', exist_ok=True)
        # cv2.drawContours(img,contours,-1,(0,0,255),1)   
        # cv2.imwrite('contour/{}'.format(f), img)
        flag = True
        for c in contours:
            for x in range(45, 55):
                for y in range(45, 55):
                    if flag and cv2.pointPolygonTest(c, (x,y), False) >= 0:
                        area = cv2.contourArea(c)
                        if area < 1500:
                            print(area)
                            area_total += area
                        flag = False
        if flag:
            print('miss')
    print(area_total)
        

if __name__ == '__main__':
    calc()
    