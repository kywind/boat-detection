import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

# a = np.load('data/SegmentationClass/Rect0_Level_18.npy')
# print(a[224, 151])

# im = np.array(Image.open('data/JPEGImages/Rect0_Level_18.jpg').convert('RGB'))
# print(im)

# import json
# with open('data/2018gt_labeled/Rect0_Level_18.json') as f:
#     a = json.load(f)
# print(a['shapes'][0]['points'])

img = cv2.imread('data_water/JPEGImages/0.jpg')
img = torch.tensor(img).permute(2, 0, 1)
jitter = transforms.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=.05)
jitted_imgs = [jitter(img) for _ in range(4)]
for i in range(4):
    cv2.imwrite('out_{}.jpg'.format(i), jitted_imgs[i].permute(1, 2, 0).numpy())
