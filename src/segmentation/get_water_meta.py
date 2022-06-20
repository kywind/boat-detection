import os
import shutil
import random
import cv2
import numpy as np


def intersect(box1, box2):  # judge if two boxes intersect
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def make_water_buffer():
    years = range(2010, 2022)
    for year in years:
        filename = '../cluster_detection/data/{}.txt'.format(year)
        outdir = 'inference/{}_water_meta/'.format(year)
        os.makedirs(outdir, exist_ok=True)

        w, h = 500, 500
        res = 0.0000107288
        w, h = w * res, h * res

        with open(filename, 'r') as f:
            data = f.read().strip().split('\n')
        aux = range(len(data))
        cnt = 0
        for i in aux:
            print(year, i, data[i])
            xmin, ymin, xmax, ymax = data[i].split()
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            box = (x - w/2, y - h/2, x + w/2, y + h/2)
            # delta_x = int((xmax - x) / res)
            # delta_y = int((ymax - y) / res)
            # # assert(delta_x > 0 and delta_y > 0)
            # img = getmap(box, year)
            # mid_x = int(img.shape[0] / 2)
            # mid_y = int(img.shape[1] / 2)
            # # img = cv2.rectangle(img, (mid_x-delta_x, mid_y-delta_y), (mid_x+delta_x, mid_y+delta_y), (0, 0, 255), 1)
            # cv2.imwrite(outdir + '{}.jpg'.format(cnt), img)
            f = open(outdir + '{}.txt'.format(cnt), 'w')
            f.write('{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
            for j in aux:
                xmin2, ymin2, xmax2, ymax2 = data[j].split()
                xmin2, ymin2, xmax2, ymax2 = eval(xmin2), eval(ymin2), eval(xmax2), eval(ymax2)
                if intersect(box, (xmin2, ymin2, xmax2, ymax2)):
                    print(xmin2, ymin2, xmax2, ymax2)
                    xmid2 = (xmin2+xmax2)/2
                    ymid2 = (ymin2+ymax2)/2
                    x2 = int((xmid2-box[0])/res)
                    y2 = int((box[3]-ymid2)/res)
                    f.write('{} {} {}\n'.format(j, x2, y2))
            f.close()
            cnt += 1


make_water_buffer()
