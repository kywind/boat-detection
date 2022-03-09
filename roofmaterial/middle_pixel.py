import os
import cv2
import numpy as np
import shutil


task = '2018_pred_single'
img_path = './result/{}/'.format(task)
label_path = None
thatch_path = './result/{}_thatch/'.format(task)
zinc_path = './result/{}_zinc/'.format(task)
if os.path.exists(thatch_path):
    shutil.rmtree(thatch_path)
if os.path.exists(zinc_path):
    shutil.rmtree(zinc_path)
os.makedirs(thatch_path, exist_ok=True)
os.makedirs(zinc_path, exist_ok=True)

img_list = [f for f in os.listdir(img_path) if f.endswith('jpg')]
for i in range(len(img_list)):
    img = cv2.imread(img_path + '%d.jpg' % i)  # bgr
    # print(img.shape)  # h, w, c=3
    h, w, _ = img.shape
    hc, wc = int(h/2), int(w/2)
    img_center = np.mean(img[hc-1:hc+2, wc-1:wc+2], axis=(0, 1))
    if img_center[0] > img_center[2]:
        # zinc
        cv2.imwrite(zinc_path + '%d.jpg' % i, img)
    else:
        # thatch
        cv2.imwrite(thatch_path + '%d.jpg' % i, img)



