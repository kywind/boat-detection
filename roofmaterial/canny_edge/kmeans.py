import cv2
import numpy as np
import os
from tqdm import tqdm


img_path='../result/2018_pred_single/'
out_path = 'grad_vis_kmeans/'
os.makedirs(out_path, exist_ok=True)
indices = range(3601)
for i in tqdm(indices):
    f = img_path + '{}.jpg'.format(i)
    img = cv2.imread(f)
    width_range = list(range(img.shape[1]))  # horizontal
    height_range = list(range(img.shape[0]))  # vertical
    for j in range(img.shape[0]):
        if img[j].mean() <= 20:
            height_range.remove(j)
    for j in range(img.shape[1]):
        if img[:, j].mean() <= 20:
            width_range.remove(j)
    img = img[height_range]
    img = img[:, width_range]
    # print(img)

    scale = 2
    width  = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    attempts = 10
    d = 1
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    dist_img = np.concatenate((img, d * xx[:, :, None], d * yy[:, :, None]), axis=2)
    flat_dist_img = dist_img.reshape((-1, 5))
    flat_dist_img = np.float32(flat_dist_img)

    ret, label, center = cv2.kmeans(flat_dist_img, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape[0], img.shape[1], 5)
    result_image = result_image[:, :, :3]
    cv2.imwrite(out_path + '{}.png'.format(i), img)
    cv2.imwrite(out_path + '{}_res.png'.format(i), result_image)