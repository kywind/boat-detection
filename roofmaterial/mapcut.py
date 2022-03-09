import shutil
import os
import random
import cv2
import sys
import numpy as np
from read_roofmaterial import read


def intersect(box1, box2):  # judge if two boxes intersect
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def getmap(box, year, ratio=1, res=None):  # get satellite map for any rectangle (source needed)
    channel = 3
    img_type = '.tif'
    img_size_bound = (6000,4000)
    map_path = '/data/rawimages/yangon_{}/'.format(year)
    tfw_path = './utils/tfw.txt'
    resolution = 0.0000107288 * ratio if not res else res
        
    ftfw = open(tfw_path, 'r')
    tfw = ftfw.read().strip().split('\n')
    ftfw.close()
    tfw_dict = dict()
    for item in tfw:
        tfw_data = item.split()
        name, xstep, ystep, x0, y0 =  tfw_data[0], eval(tfw_data[1]), eval(tfw_data[2]), eval(tfw_data[3]), eval(tfw_data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    
    x_min, y_min, x_max, y_max = box
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    res = np.zeros((height,width,channel))
    # hlist, wlist = [], []
    for name in tfw_dict.keys():
        xstep, ystep, x0, y0 = tfw_dict[name]
        x1, y1 = x0 + img_size_bound[0] * xstep, y0 + img_size_bound[1] * ystep
        if not os.path.exists(map_path + name + '_Level_17' + img_type):
            continue
        if intersect((x0,y1,x1,y0), box):
            print(name)
            img = cv2.imread(map_path + name + '_Level_17' + img_type)
            # print(img.shape, xstep, ystep)
            i_list = [i for i in range(img.shape[1]) if x0 + i * xstep >= x_min and x0 + i * xstep < x_max and 
                      (x0 + i * xstep - x_min) % resolution >= 0 and (x0 + i * xstep - x_min) % resolution < abs(xstep)]
            j_list = [j for j in range(img.shape[0]) if y0 + j * ystep >= y_min and y0 + j * ystep < y_max and 
                      (y0 + j * ystep - y_min) % resolution >= 0 and (y0 + j * ystep - y_min) % resolution < abs(ystep)]
            # print(xstep, resolution, len(i_list), len(j_list))
            ii, jj = np.meshgrid(i_list, j_list)
            ii = ii.reshape(-1)
            jj = jj.reshape(-1)
            deltax = x0 - x_min
            deltay = y0 - y_min
            for t in range(ii.shape[0]):
                i, j = ii[t], jj[t]
                h = int((deltay + j * ystep) / resolution)
                w = int((deltax + i * xstep) / resolution)
                if h >= res.shape[0] or w >= res.shape[1]:
                    continue 
                # if w not in wlist:
                #     wlist.append(w)
                # if h not in hlist:
                #     hlist.append(h)
                res[h][w] = img[j][i]
                # res[h][w] = np.array([255,255,255])
    # print(len(wlist), width, len(hlist), height)
    # assert(len(wlist) == width and len(hlist) == height)
    res = cv2.flip(res, 0)
    return res


def mapcut_single(year):
    filename = './data/{}.txt'.format(year)
    savepath = './result/{}_single_png_1000/'.format(year)

    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.mkdir(savepath)
    os.makedirs(savepath, exist_ok=True)

    cnt = 0
    w, h = 500, 500
    w, h = w * 0.0000107288, h * 0.0000107288
    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')
        for i in range(len(data)):
            print(i)
            print(data[i])
            xmin, ymin, xmax, ymax = data[i].split()
            xmin, ymin, xmax, ymax = eval(xmin)-w/2, eval(ymin)-h/2, eval(xmax)+w/2, eval(ymax)+h/2
            img = getmap((xmin, ymin, xmax, ymax), 2018)
            cv2.imwrite(savepath + '{}.png'.format(cnt), img)
            f = open(savepath + '{}.txt'.format(cnt), 'w')
            f.write('{} {} {} {}'.format(xmin, ymin, xmax, ymax))
            f.close()
            cnt += 1


def label_check():
    savepath = './result/label_check/'
    w, h = 1000, 1000
    w, h = w * 0.0000107288, h * 0.0000107288

    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.mkdir(savepath)
    os.makedirs(savepath, exist_ok=True)

    cnt = 0
    year = 2018
    interview_keys, latitudes, longtitudes, zincs, thatchs, house_nums = read()
    for i in range(len(interview_keys)):
        x, y = longtitudes[i], latitudes[i]
        box = (x-w/2, y-h/2, x+w/2, y+h/2)
        img = getmap(box, year)
        cv2.imwrite(savepath + '{}_{}.png'.format(cnt, 'zinc' if zincs[i]==1 else 'thatch'), img)
        f = open(savepath + '{}.txt'.format(cnt), 'w')
        f.write('{} {} {} {}'.format(interview_keys[i], latitudes[i], longtitudes[i], house_nums[i]))
        f.close()
        cnt += 1



if __name__ == '__main__':
    mapcut_single(2018)
    # label_check()
                        