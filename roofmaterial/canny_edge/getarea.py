import cv2
import numpy as np
import os
from tqdm import tqdm


img_path = 'grad_vis_new_png_png/'
out_path = 'res_new/'
os.makedirs(out_path, exist_ok=True)

indices = range(3601)
total_size = 0
scale = 4
for i in tqdm(indices):
    f = img_path + '{}_grad.png'.format(i)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    f_orig = img_path + '{}.png'.format(i)
    img_orig = cv2.imread(f_orig)
    mask = -img.copy()
    mask[mask == 255] = 1
    cnt = 2
    while np.any(mask == 0):
        seed_x = np.where(mask == 0)[1][0]
        seed_y = np.where(mask == 0)[0][0]
        cv2.floodFill(mask, None, (seed_x, seed_y), cnt)
        cnt += 1

    min_weight = 999999
    min_n = -1
    final_size = 0
    img_center = np.array([img.shape[0]/2, img.shape[1]/2])
    corners = np.array([[0,0], [0,img.shape[1]-1], [img.shape[0]-1,0], [img.shape[0]-1, img.shape[1]-1]])
    img_size = img.shape[0] * img.shape[1]
    for n in range(2, cnt):
        indices = np.where(mask == n)
        indices = np.array([indices[0], indices[1]]).astype(np.float32).T
        weight = (np.abs(indices - img_center)).sum(axis=1).mean()
        size = indices.shape[0]
        corner_count = 0
        for k in range(4):
            if mask[corners[k,0], corners[k,1]] == n:
                corner_count += 1
        if weight < min_weight and size >= 0.05 * img_size and size <= 0.7 * img_size and corner_count < 3:
            min_n = n
            min_weight = weight
            final_size = size
    
    # mask = mask[1:-1, 1:-1]
    mask = 0. * (mask != min_n) + 1. * (mask == min_n)
    total_size += final_size
    img_out = img_orig * mask[:, :, None]
    img_out = img_out.astype(np.uint8)
    cv2.imwrite(out_path + '{}.png'.format(i), img_orig)
    # cv2.imwrite(out_path + '{}_grad.png'.format(i), img)
    cv2.imwrite(out_path + '{}_res.png'.format(i), img_out)
print(total_size / scale / scale, total_size * 0.0000107288 * 0.0000107288 * 110000 * 110000 / scale / scale)




    