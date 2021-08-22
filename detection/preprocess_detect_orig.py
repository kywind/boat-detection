import os, sys
import random

imgpath = '/mnt/satellite/JPEGImages/JPEGImages_2010/'
saveBasePath = 'imglist/detect.txt'

total_jpg = os.listdir(imgpath)
ftest = open(saveBasePath, 'w')
count = 0
for jpg in total_jpg:
    name = imgpath + jpg + '\n'
    ftest.write(name)
    count += 1
print('Get {} images to detect.'.format(count))
ftest.close()