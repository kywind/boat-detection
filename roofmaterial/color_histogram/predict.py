import os, shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from model import Net


def main():
    device = 'cuda'
    img_path = '../result/2018_pred_single/'
    imgs = [f for f in os.listdir(img_path) if f.endswith('jpg')][:100]
    indices = range(len(imgs))
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    kmeans = KMeans(n_clusters=4, random_state=0)

    his_all = []
    for i in tqdm(indices):
        f = img_path + '{}.jpg'.format(i)
        img = np.array(cv2.imread(f)).astype('float32').reshape(-1, 3)
        his = kmeans.fit(img).cluster_centers_
        his_all.append(his)
    his_all = np.stack(his_all, axis=0) / 255.

    data = transform(his_all).unsqueeze(3).transpose(1, 0).to(device)

    model = Net().to(device)
    model.load_state_dict(torch.load('ckpt.pt'))

    output = model(data)

    thatch_path = './result/thatch/'
    zinc_path = './result/zinc/'
    if os.path.exists(thatch_path):
        shutil.rmtree(thatch_path)
    if os.path.exists(zinc_path):
        shutil.rmtree(zinc_path)
    os.makedirs(thatch_path, exist_ok=True)
    os.makedirs(zinc_path, exist_ok=True)

    for i in tqdm(indices):
        if output[i].item() > 0.5:  # zinc
            img = cv2.imread(img_path + '{}.jpg'.format(i))
            cv2.imwrite(zinc_path + '{}.jpg'.format(i), img)
        else:
            img = cv2.imread(img_path + '{}.jpg'.format(i))
            cv2.imwrite(thatch_path + '{}.jpg'.format(i), img)


if __name__ == '__main__':
    main()
