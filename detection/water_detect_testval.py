import os, sys
import random
import xml.etree.ElementTree as ET
import cv2 
import numpy as np
import shutil
import time
from tqdm import tqdm
from water_detect import get_water

task = 'val'
orig_imglist = 'imglist/{}.txt'.format(task)
nowater_imglist = 'imglist/{}_nowater.txt'.format(task)
haswater_imglist = 'imglist/{}_haswater.txt'.format(task)

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