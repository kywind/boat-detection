import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import cv2
from tqdm import tqdm


######################## EXAMPLE ##########################
# >>> import numpy as np
# >>> from sklearn.linear_model import LinearRegression
# >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# >>> # y = 1 * x_0 + 2 * x_1 + 3
# >>> y = np.dot(X, np.array([1, 2])) + 3
# >>> reg = LinearRegression().fit(X, y)
# >>> reg.score(X, y)
# 1.0
# >>> reg.coef_
# array([1., 2.])
# >>> reg.intercept_
# 3.0...
# >>> reg.predict(np.array([[3, 5]]))
# array([16.])
###########################################################


img_path='../result/2018_pred_single/'
label_path='./label.txt'
        
with open(label_path) as f:
    label = f.read().strip().split()
    label_thatch = label[1].strip().split(',')
    label_zinc = label[3].strip().split(',')
    indices = label_thatch + label_zinc
    size = len(indices)
        
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

kmeans = KMeans(n_clusters=5, random_state=0)

his_all = []
for i in indices:
    f = img_path + '{}.jpg'.format(i)
    img = np.array(cv2.imread(f)).astype('float32').reshape(-1, 3)
    his = kmeans.fit(img).cluster_centers_
    his_all.append(his)
his_all = np.stack(his_all, axis=0) / 255.

data = transform(his_all).unsqueeze(3).transpose(1, 0)
label = torch.zeros(size)
label[int(size/2):] = 1
print(data.shape, label.shape)
