import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
import cv2
from tqdm import tqdm

##################### EXAMPLE ##########################
# >>> X = np.array([[1, 2], [1, 4], [1, 0],
# ...               [10, 2], [10, 4], [10, 0]])
# >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# >>> kmeans.labels_
# array([1, 1, 1, 0, 0, 0], dtype=int32)
# >>> kmeans.predict([[0, 0], [12, 3]])
# array([1, 0], dtype=int32)
# >>> kmeans.cluster_centers_
# array([[10.,  2.],
#        [ 1.,  2.]])
########################################################

class ColorHistogramDataset(Dataset):
    def __init__(self, img_path='../result/2018_pred_single/', 
                       label_path='./label.txt'):
        
        with open(label_path) as f:
            label = f.read().strip().split()
            label_thatch = label[1].strip().split(',')
            label_zinc = label[3].strip().split(',')
            # assert(len(label_thatch) == 48 and len(label_zinc) == 48)
            for index in label_zinc:
                os.system('cp ../result/2018_pred_single/{}.jpg ./label_zinc/'.format(index))
                os.system('cp ../result/2018_pred_single/{}.txt ./label_zinc/'.format(index))
            raise Exception
            indices = label_thatch + label_zinc
            self.size = len(indices)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        kmeans = KMeans(n_clusters=4, random_state=0)

        his_all = []
        for i in indices:
            f = img_path + '{}.jpg'.format(i)
            img = np.array(cv2.imread(f)).astype('float32').reshape(-1, 3)
            his = kmeans.fit(img).cluster_centers_
            his_all.append(his)
        his_all = np.stack(his_all, axis=0) / 255.

        self.data = transform(his_all).unsqueeze(3).transpose(1, 0)
        self.label = torch.zeros(self.size)
        self.label[int(self.size/2):] = 1
        print(self.data.shape, self.label.shape)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # data = self.data[idx]
        # assert(data.shape[1] == 5)
        # data = data[:, torch.randperm(5)]
        # return data, self.label[idx]
        return self.data[idx], self.label[idx]


if __name__ == '__main__':
    test = ColorHistogramDataset()